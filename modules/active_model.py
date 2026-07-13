import torch
import torch.nn.functional as F
from modules.HDC_utils import EllipsoidModel

class ActiveModel(EllipsoidModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize storage for Oracle Subclusters
        self.register_buffer('oracle_subclusters', torch.empty((0, self.hd_dim), dtype=torch.float32, device=self.device))
        self.register_buffer('oracle_subcluster_labels', torch.empty((0,), dtype=torch.long, device=self.device))

    def inference_update_ooa(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Density-Filtered Outliers (Outlier Oracle Anchor) Active Domain Adaptation"""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                
            num_active = enc_norm.shape[0]
            
            # Predictions logic
            # Use oracle subclusters if they exist
            chunk_logits = self.classify(enc_norm)
            preds = torch.argmax(chunk_logits, dim=1)
            
            # Removed hard override prediction logic to maintain smooth Voronoi boundaries
                
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

            # Active domain adaptation logic
            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                
                # 1. Density filter
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    # number of neighbors within 0.5m radius
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                # 2. Highest distance from known subclusters
                if torch.any(valid_density_mask):
                    subclusters = F.normalize(self.subclusters.data).to(enc_norm.dtype)
                    sims_sub = enc_norm @ subclusters.T
                    max_sims_sub, _ = sims_sub.max(dim=1)
                    
                    # Also consider distance to existing oracle subclusters
                    if self.oracle_subclusters.shape[0] > 0:
                        sims_oracle_sub = enc_norm @ F.normalize(self.oracle_subclusters).T.to(enc_norm.dtype)
                        max_sims_oracle_sub, _ = sims_oracle_sub.max(dim=1)
                        max_sims_sub = torch.maximum(max_sims_sub, max_sims_oracle_sub)
                    
                    # Ignore points with invalid density by setting their similarity very high
                    max_sims_sub[~valid_density_mask] = float('inf')
                    
                    # Point with the highest distance (lowest max similarity)
                    outlier_idx = max_sims_sub.argmin()
                    
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        # Register new oracle subcluster
                        new_subcluster = enc_norm[outlier_idx].unsqueeze(0)
                        self.oracle_subclusters = torch.cat([self.oracle_subclusters, new_subcluster], dim=0)
                        self.oracle_subcluster_labels = torch.cat([self.oracle_subcluster_labels, torch.tensor([oracle_label], device=self.device)])
                        
                        # Soft HDC Integration: bundle it directly into the target class's prototype with standard mass weighting
                        self.classify.weight[oracle_label] = F.normalize(
                            self.classify.weight[oracle_label] + new_subcluster.squeeze(0), dim=0
                        )
                        
            # Standard pull update
            selected_proto = F.normalize(self.classify.weight[preds])
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_oagp(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Oracle-Anchored Graph Propagation (OAGP)"""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                
            prototypes = F.normalize(self.classify.weight)
            S = enc_norm @ prototypes.T  # Shape: (num_active, num_classes)
            
            has_oracle = False
            outlier_idx = -1
            oracle_label = -1
            
            # ADA Outlier Finding Logic
            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                num_active = enc_norm.shape[0]
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                if torch.any(valid_density_mask):
                    subclusters = F.normalize(self.subclusters.data).to(enc_norm.dtype)
                    sims_sub = enc_norm @ subclusters.T
                    max_sims_sub, _ = sims_sub.max(dim=1)
                    max_sims_sub[~valid_density_mask] = float('inf')
                    
                    outlier_idx = max_sims_sub.argmin()
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        has_oracle = True
                        S[outlier_idx, :] = 0.0
                        S[outlier_idx, oracle_label] = 1.0

            if proj_xyz is not None:
                # Same graph logic as inference_update_gplp
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                num_active = active_xyz.shape[0]
                K = 5
                iterations = 3
                
                topk_indices = []
                chunk_s = 5000
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    _, knn_idx = dist.topk(K + 1, largest=False)
                    topk_indices.append(knn_idx[:, 1:]) 
                knn_idx = torch.cat(topk_indices, dim=0) 
                
                for _ in range(iterations):
                    neighbor_scores = S[knn_idx]
                    S = 0.5 * S + 0.5 * neighbor_scores.mean(dim=1)
                    if has_oracle:
                        S[outlier_idx, :] = 0.0
                        S[outlier_idx, oracle_label] = 1.0
            
            predictions = S.argmax(dim=1)
            
            smoothed_sims = S.gather(1, predictions.unsqueeze(1)).squeeze(1)
            distances = (1.0 - smoothed_sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = predictions

            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(predictions[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (predictions == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_mvos(self, x, beta=0.2, distance_sensitivity=1.0, learning_rate=0.01, chunk_size=-1, max_updates_per_class=-1, thresholds=[0.45, 0.80], oracle_labels=None, proj_xyz=None):
        """Multi-View Oracle Subclustering (MVOS)"""
        self.eval()
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            original_x = x.permute(0, 2, 3, 1).contiguous().reshape(-1, x.shape[1])
            valid_enc_mask = torch.any(original_x != 0, dim=1)
            
            if not torch.any(valid_enc_mask):
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            
            active_enc = enc[valid_enc_mask]
            enc_norm = F.normalize(active_enc)
            if enc_norm.dtype != self.classify.weight.dtype:
                enc_norm = enc_norm.to(self.classify.weight.dtype)
                
            num_active = enc_norm.shape[0]
            chunk_logits = self.classify(enc_norm)
            preds = torch.argmax(chunk_logits, dim=1)
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                if torch.any(valid_density_mask):
                    subclusters = F.normalize(self.subclusters.data).to(enc_norm.dtype)
                    sims_sub = enc_norm @ subclusters.T
                    max_sims_sub, _ = sims_sub.max(dim=1)
                    max_sims_sub[~valid_density_mask] = float('inf')
                    
                    outlier_idx = max_sims_sub.argmin()
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        # 2. Apply TTAug specifically to this Oracle point (Generate 5 views)
                        x_aug1 = torch.roll(x, shifts=14, dims=3)
                        x_aug2 = torch.roll(x, shifts=-14, dims=3)
                        x_aug3 = x * 1.05
                        x_aug4 = x * 0.95
                        x_aug5 = x + torch.randn_like(x) * 0.05
                        
                        enc1, _, _ = self.encode(x_aug1)
                        enc2, _, _ = self.encode(x_aug2)
                        enc3, _, _ = self.encode(x_aug3)
                        enc4, _, _ = self.encode(x_aug4)
                        enc5, _, _ = self.encode(x_aug5)
                        
                        # Extract the outlier's hypervector for all 5 views + original
                        h_views = [
                            F.normalize(enc[valid_enc_mask][outlier_idx].unsqueeze(0)),
                            F.normalize(enc1[valid_enc_mask][outlier_idx].unsqueeze(0)),
                            F.normalize(enc2[valid_enc_mask][outlier_idx].unsqueeze(0)),
                            F.normalize(enc3[valid_enc_mask][outlier_idx].unsqueeze(0)),
                            F.normalize(enc4[valid_enc_mask][outlier_idx].unsqueeze(0)),
                            F.normalize(enc5[valid_enc_mask][outlier_idx].unsqueeze(0))
                        ]
                        
                        for h in h_views:
                            self.classify.weight[oracle_label] = F.normalize(
                                self.classify.weight[oracle_label] + h.squeeze(0), dim=0
                            )
                        
            # Standard pull update
            selected_proto = F.normalize(self.classify.weight[preds])
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_fcsr(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, distance_sensitivity=3.0, thresholds=[0.45, 0.80], beta=0.2, max_updates_per_class=-1):
        with torch.no_grad():
            enc, indices, is_wrong_left = self.encode(x, PERCENTAGE=None, is_wrong=None)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
            S = enc_norm @ prototypes.T
            preds = S.argmax(dim=1)
            
            # ADA Outlier Finding Logic
            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                num_active = enc_norm.shape[0]
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                if torch.any(valid_density_mask):
                    subclusters = F.normalize(self.subclusters.data).to(enc_norm.dtype)
                    sims_sub = enc_norm @ subclusters.T
                    max_sims_sub, _ = sims_sub.max(dim=1)
                    max_sims_sub[~valid_density_mask] = float('inf')
                    
                    outlier_idx = max_sims_sub.argmin()
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        x_aug1 = torch.roll(x, shifts=14, dims=3)
                        x_aug2 = x * 1.05
                        enc1, _, _ = self.encode(x_aug1)
                        enc2, _, _ = self.encode(x_aug2)
                        
                        bundle = F.normalize(enc[valid_enc_mask][outlier_idx].unsqueeze(0) + 
                                             enc1[valid_enc_mask][outlier_idx].unsqueeze(0) + 
                                             enc2[valid_enc_mask][outlier_idx].unsqueeze(0))
                        
                        if not hasattr(self, 'fcsr_ages'):
                            self.fcsr_ages = torch.zeros(self.subclusters.shape[0], device=self.device)
                            self.fcsr_counts = torch.bincount(self.subcluster_to_class, minlength=self.num_classes)
                            self.fcsr_capacity = 20
                        
                        self.fcsr_ages += 1
                        
                        if self.fcsr_counts[oracle_label] < self.fcsr_capacity:
                            self.subclusters = torch.nn.Parameter(torch.cat([self.subclusters.data, bundle.to(self.subclusters.dtype)], dim=0))
                            self.subcluster_to_class = torch.cat([self.subcluster_to_class, torch.tensor([oracle_label], device=self.device)])
                            self.fcsr_ages = torch.cat([self.fcsr_ages, torch.tensor([0.0], device=self.device)])
                            self.fcsr_counts[oracle_label] += 1
                        else:
                            class_mask = (self.subcluster_to_class == oracle_label)
                            class_indices = torch.nonzero(class_mask).squeeze(1)
                            if len(class_indices) > 0:
                                oldest_idx = class_indices[self.fcsr_ages[class_indices].argmax()]
                                self.subclusters.data[oldest_idx] = bundle.squeeze(0).to(self.subclusters.dtype)
                                self.fcsr_ages[oldest_idx] = 0.0

            # Rest of method is standard pull update...
            selected_proto = F.normalize(self.classify.weight[preds])
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_owmp(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, distance_sensitivity=3.0, thresholds=[0.45, 0.80], beta=0.2, max_updates_per_class=-1):
        with torch.no_grad():
            enc, indices, is_wrong_left = self.encode(x, PERCENTAGE=None, is_wrong=None)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
            S = enc_norm @ prototypes.T
            preds = S.argmax(dim=1)
            
            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                num_active = enc_norm.shape[0]
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                if torch.any(valid_density_mask):
                    subclusters = F.normalize(self.subclusters.data).to(enc_norm.dtype)
                    sims_sub = enc_norm @ subclusters.T
                    max_sims_sub, _ = sims_sub.max(dim=1)
                    max_sims_sub[~valid_density_mask] = float('inf')
                    
                    outlier_idx = max_sims_sub.argmin()
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        x_aug1 = torch.roll(x, shifts=14, dims=3)
                        x_aug2 = x * 1.05
                        enc1, _, _ = self.encode(x_aug1)
                        enc2, _, _ = self.encode(x_aug2)
                        
                        bundle = F.normalize(enc[valid_enc_mask][outlier_idx].unsqueeze(0) + 
                                             enc1[valid_enc_mask][outlier_idx].unsqueeze(0) + 
                                             enc2[valid_enc_mask][outlier_idx].unsqueeze(0)).squeeze(0)
                        
                        # Direct EMA Pull to Master Prototype
                        lr_oracle = 0.05
                        self.classify.weight[oracle_label] = F.normalize(
                            (1 - lr_oracle) * self.classify.weight[oracle_label] + lr_oracle * bundle.to(self.classify.weight.dtype), dim=0
                        )

            selected_proto = F.normalize(self.classify.weight[preds])
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_mgoa(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, distance_sensitivity=3.0, thresholds=[0.45, 0.80], beta=0.2, max_updates_per_class=-1):
        with torch.no_grad():
            enc, indices, is_wrong_left = self.encode(x, PERCENTAGE=None, is_wrong=None)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
            S = enc_norm @ prototypes.T
            preds = S.argmax(dim=1)
            
            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                num_active = enc_norm.shape[0]
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                if torch.any(valid_density_mask):
                    # Margin gating: find point with smallest margin between Top-1 and Top-2
                    top2_sims, _ = S[valid_density_mask].topk(2, dim=1)
                    margin = top2_sims[:, 0] - top2_sims[:, 1]
                    min_margin_idx_in_dense = margin.argmin()
                    
                    valid_indices = torch.nonzero(valid_density_mask).squeeze(1)
                    outlier_idx = valid_indices[min_margin_idx_in_dense]
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        x_aug1 = torch.roll(x, shifts=14, dims=3)
                        x_aug2 = x * 1.05
                        enc1, _, _ = self.encode(x_aug1)
                        enc2, _, _ = self.encode(x_aug2)
                        
                        bundle = F.normalize(enc[valid_enc_mask][outlier_idx].unsqueeze(0) + 
                                             enc1[valid_enc_mask][outlier_idx].unsqueeze(0) + 
                                             enc2[valid_enc_mask][outlier_idx].unsqueeze(0))
                        
                        self.subclusters = torch.nn.Parameter(torch.cat([self.subclusters.data, bundle.to(self.subclusters.dtype)], dim=0))
                        self.subcluster_to_class = torch.cat([self.subcluster_to_class, torch.tensor([oracle_label], device=self.device)])

            selected_proto = F.normalize(self.classify.weight[preds])
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_vgo(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, distance_sensitivity=3.0, thresholds=[0.45, 0.80], beta=0.2, max_updates_per_class=-1):
        with torch.no_grad():
            enc, indices, is_wrong_left = self.encode(x, PERCENTAGE=None, is_wrong=None)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
            S = enc_norm @ prototypes.T
            preds = S.argmax(dim=1)
            
            if oracle_labels is not None and proj_xyz is not None:
                active_oracle_labels = oracle_labels.view(-1)[valid_enc_mask]
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                num_active = enc_norm.shape[0]
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 15
                valid_density_mask = densities >= density_threshold
                
                if torch.any(valid_density_mask):
                    subclusters = F.normalize(self.subclusters.data).to(enc_norm.dtype)
                    sims_sub = enc_norm @ subclusters.T
                    max_sims_sub, _ = sims_sub.max(dim=1)
                    max_sims_sub[~valid_density_mask] = float('inf')
                    
                    outlier_idx = max_sims_sub.argmin()
                    oracle_label = active_oracle_labels[outlier_idx]
                    
                    if oracle_label >= 0 and oracle_label < self.num_classes:
                        x_aug1 = torch.roll(x, shifts=14, dims=3)
                        x_aug2 = x * 1.05
                        enc1, _, _ = self.encode(x_aug1)
                        enc2, _, _ = self.encode(x_aug2)
                        
                        base_h = enc[valid_enc_mask][outlier_idx].unsqueeze(0)
                        roll_h = enc1[valid_enc_mask][outlier_idx].unsqueeze(0)
                        scale_h = enc2[valid_enc_mask][outlier_idx].unsqueeze(0)
                        
                        sim = F.cosine_similarity(base_h, roll_h).item()
                        if sim >= 0.50:
                            bundle = F.normalize(base_h + roll_h + scale_h)
                            self.subclusters = torch.nn.Parameter(torch.cat([self.subclusters.data, bundle.to(self.subclusters.dtype)], dim=0))
                            self.subcluster_to_class = torch.cat([self.subcluster_to_class, torch.tensor([oracle_label], device=self.device)])

            selected_proto = F.normalize(self.classify.weight[preds])
            sims = torch.sum(enc_norm * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds

            if not torch.any(update_mask):
                return full_predictions

            valid_indices_in_active = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices_in_active])

            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                class_indices = torch.nonzero(class_mask).squeeze(1)

                if max_updates_per_class != -1 and len(class_indices) > max_updates_per_class:
                    fps_indices = self._farthest_point_sample(enc_norm[class_indices].cpu(), max_updates_per_class)
                    class_indices = class_indices[fps_indices.to(self.device)]

                sample_encs = enc_norm[class_indices]
                sub_sims, _ = self.get_max_subcluster_similarity(sample_encs, c_id, distance_sensitivity)

                valid_mask = sub_sims > thresholds[0]
                if not torch.any(valid_mask):
                    continue

                sample_encs = sample_encs[valid_mask]
                sub_sims = sub_sims[valid_mask]

                weights = sub_sims / sub_sims.sum()
                weighted_pull_vector = (sample_encs * weights.unsqueeze(1)).sum(dim=0)
                effective_lr = learning_rate * sub_sims.mean().item()

                current_weight = self.classify.weight[c_id]
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * weighted_pull_vector
                updated_weight = (1.0 - effective_lr) * current_weight + effective_lr * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)

            return full_predictions

    def inference_update_cws(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, gamma=2.0, max_updates_per_class=-1, **kwargs):
        with torch.no_grad():
            enc, indices, is_wrong_left = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
            
            S = enc_norm @ prototypes.T
            preds = S.argmax(dim=1)
            
            x_aug1 = torch.roll(x, shifts=14, dims=3)
            x_aug2 = x * 1.05
            
            enc1, _, _ = self.encode(x_aug1)
            enc2, _, _ = self.encode(x_aug2)
            
            enc1_norm = F.normalize(enc1[valid_enc_mask])
            enc2_norm = F.normalize(enc2[valid_enc_mask])
            
            if enc1_norm.dtype != prototypes.dtype:
                enc1_norm = enc1_norm.to(prototypes.dtype)
            if enc2_norm.dtype != prototypes.dtype:
                enc2_norm = enc2_norm.to(prototypes.dtype)
                
            selected_proto = prototypes[preds]
            
            w0 = torch.sum(enc_norm * selected_proto, dim=1).clamp(min=0.0)
            w1 = torch.sum(enc1_norm * selected_proto, dim=1).clamp(min=0.0)
            w2 = torch.sum(enc2_norm * selected_proto, dim=1).clamp(min=0.0)
            
            w0 = (w0 ** gamma).unsqueeze(1)
            w1 = (w1 ** gamma).unsqueeze(1)
            w2 = (w2 ** gamma).unsqueeze(1)
            
            bundled_enc = F.normalize((w0 * enc_norm) + (w1 * enc1_norm) + (w2 * enc2_norm))
            
            sims = torch.sum(bundled_enc * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                sample_encs = bundled_enc[class_mask]
                
                pull_vector = sample_encs.mean(dim=0)
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
                updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_dava(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, **kwargs):
        with torch.no_grad():
            enc, indices, is_wrong_left = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
                
            S = enc_norm @ prototypes.T
            preds = S.argmax(dim=1)
            
            bundled_enc = enc_norm.clone()
            
            if proj_xyz is not None:
                xyz_flat = proj_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
                active_xyz = xyz_flat[valid_enc_mask]
                
                chunk_s = 5000
                num_active = active_xyz.shape[0]
                densities = torch.zeros(num_active, device=self.device)
                for i in range(0, num_active, chunk_s):
                    end = min(i + chunk_s, num_active)
                    chunk_xyz = active_xyz[i:end]
                    dist = torch.cdist(chunk_xyz, active_xyz)
                    densities[i:end] = (dist < 0.5).sum(dim=1).float()
                
                density_threshold = 5 
                is_sparse = densities < density_threshold
                is_dense = ~is_sparse
                
                if torch.any(is_dense):
                    x_yaw = torch.roll(x, shifts=14, dims=3)
                    enc_yaw, _, _ = self.encode(x_yaw)
                    enc_yaw_norm = F.normalize(enc_yaw[valid_enc_mask][is_dense])
                    if enc_yaw_norm.dtype != prototypes.dtype:
                        enc_yaw_norm = enc_yaw_norm.to(prototypes.dtype)
                    bundled_enc[is_dense] = F.normalize(enc_norm[is_dense] + enc_yaw_norm)
                
                if torch.any(is_sparse):
                    x_yaw = torch.roll(x, shifts=14, dims=3)
                    x_scale = x * 1.05
                    x_jitter = x + torch.randn_like(x) * 0.01
                    x_drop = F.dropout(x, p=0.1)
                    
                    enc_yaw, _, _ = self.encode(x_yaw)
                    enc_scale, _, _ = self.encode(x_scale)
                    enc_jitter, _, _ = self.encode(x_jitter)
                    enc_drop, _, _ = self.encode(x_drop)
                    
                    e_yaw = F.normalize(enc_yaw[valid_enc_mask][is_sparse]).to(prototypes.dtype)
                    e_scale = F.normalize(enc_scale[valid_enc_mask][is_sparse]).to(prototypes.dtype)
                    e_jitter = F.normalize(enc_jitter[valid_enc_mask][is_sparse]).to(prototypes.dtype)
                    e_drop = F.normalize(enc_drop[valid_enc_mask][is_sparse]).to(prototypes.dtype)
                    
                    bundled_enc[is_sparse] = F.normalize(
                        enc_norm[is_sparse] + e_yaw + e_scale + e_jitter + e_drop
                    )
            
            selected_proto = prototypes[preds]
            sims = torch.sum(bundled_enc * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                sample_encs = bundled_enc[class_mask]
                
                pull_vector = sample_encs.mean(dim=0)
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
                updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_mssb(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, **kwargs):
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
                
            S = enc_norm @ prototypes.T
            preds = S.argmax(dim=1)
            
            x_small = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
            x_large = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
            
            enc_small, _, _ = self.encode(x_small)
            enc_large, _, _ = self.encode(x_large)
            
            enc_small_norm = F.normalize(enc_small[valid_enc_mask])
            enc_large_norm = F.normalize(enc_large[valid_enc_mask])
            
            if enc_small_norm.dtype != prototypes.dtype:
                enc_small_norm = enc_small_norm.to(prototypes.dtype)
            if enc_large_norm.dtype != prototypes.dtype:
                enc_large_norm = enc_large_norm.to(prototypes.dtype)
                
            bundled_enc = F.normalize(enc_norm + enc_small_norm + enc_large_norm)
            
            selected_proto = prototypes[preds]
            sims = torch.sum(bundled_enc * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                sample_encs = bundled_enc[class_mask]
                
                pull_vector = sample_encs.mean(dim=0)
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
                updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions

    def inference_update_tha(self, x, oracle_labels=None, proj_xyz=None, learning_rate=0.001, beta=0.2, alpha=0.5, **kwargs):
        with torch.no_grad():
            enc, _, _ = self.encode(x)
            num_total_samples = enc.shape[0]
            valid_enc_mask = (enc.abs().sum(dim=1) > 0)
            enc_norm = F.normalize(enc[valid_enc_mask])
            
            if enc_norm.shape[0] == 0:
                return torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
                
            prototypes = F.normalize(self.classify.weight)
            if enc_norm.dtype != prototypes.dtype:
                enc_norm = enc_norm.to(prototypes.dtype)
                
            if not hasattr(self, 'tha_memory'):
                self.tha_memory = torch.zeros((num_total_samples, self.hd_dim), device=self.device, dtype=prototypes.dtype)
                
            self.tha_memory[valid_enc_mask] = F.normalize(
                alpha * self.tha_memory[valid_enc_mask] + (1 - alpha) * enc_norm
            )
            
            bundled_enc = F.normalize(self.tha_memory[valid_enc_mask])
            
            S = bundled_enc @ prototypes.T
            preds = S.argmax(dim=1)
            
            selected_proto = prototypes[preds]
            sims = torch.sum(bundled_enc * selected_proto, dim=1)
            distances = (1.0 - sims) / 2.0
            update_mask = distances > beta
            
            full_predictions = torch.zeros(num_total_samples, device=self.device, dtype=torch.long)
            full_predictions[valid_enc_mask] = preds
            
            if not torch.any(update_mask):
                return full_predictions

            valid_indices = torch.nonzero(update_mask).squeeze(1)
            unique_classes = torch.unique(preds[valid_indices])
            
            for class_id in unique_classes:
                c_id = class_id.item()
                class_mask = (preds == c_id) & update_mask
                sample_encs = bundled_enc[class_mask]
                
                pull_vector = sample_encs.mean(dim=0)
                self.proto_momentum[c_id] = 0.9 * self.proto_momentum[c_id] + 0.1 * pull_vector
                updated_weight = (1.0 - learning_rate) * self.classify.weight[c_id] + learning_rate * self.proto_momentum[c_id]
                self.classify.weight[c_id] = F.normalize(updated_weight.unsqueeze(0), dim=1).squeeze(0)
                
            return full_predictions
