"""Two-tower (dual encoder) recommendation model.

CustomerTower encodes customer features into a 256-dim embedding.
ProductTower encodes product features into a 256-dim embedding.
Purchase affinity = dot product of normalized embeddings.

Enhanced with behavioral features (spend patterns, coupon engagement)
and margin-weighted loss for revenue optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomerTower(nn.Module):
    """Encodes customer features into a 256-dim L2-normalized embedding.

    Input features (89 dims total):
        customer_id  -> Embedding(10M+1, 64)           = 64
        age          -> normalized float                = 1
        gender       -> one-hot (F, M, NB)              = 3
        state        -> Embedding(51, 16)               = 16
        is_student   -> binary                          = 1
        total_spend  -> normalized float                = 1
        coupon_engagement_score -> float                = 1
        coupon_redemption_rate  -> float                = 1
        avg_basket_size -> normalized float             = 1
    """

    def __init__(self, num_customers: int = 10_000_001, num_states: int = 51,
                 cust_embed_dim: int = 64, state_embed_dim: int = 16,
                 hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.customer_embed = nn.Embedding(num_customers, cust_embed_dim)
        self.state_embed = nn.Embedding(num_states, state_embed_dim)
        # 64 + 1 + 3 + 16 + 1 + 1 + 1 + 1 + 1 = 89
        input_dim = cust_embed_dim + 1 + 3 + state_embed_dim + 1 + 1 + 1 + 1 + 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, customer_id, age, gender_onehot, state_id,
                is_student, total_spend, coupon_engagement,
                coupon_redemption_rate, avg_basket_size):
        ce = self.customer_embed(customer_id)       # (B, 64)
        se = self.state_embed(state_id)             # (B, 16)
        x = torch.cat([
            ce,
            age.unsqueeze(1),
            gender_onehot,
            se,
            is_student.unsqueeze(1),
            total_spend.unsqueeze(1),
            coupon_engagement.unsqueeze(1),
            coupon_redemption_rate.unsqueeze(1),
            avg_basket_size.unsqueeze(1),
        ], dim=1)                                   # (B, 89)
        x = F.relu(self.fc1(x))                     # (B, 256)
        x = self.fc2(x)                             # (B, 256)
        x = F.normalize(x, p=2, dim=1)              # L2 normalize
        return x


class ProductTower(nn.Module):
    """Encodes product features into a 256-dim L2-normalized embedding.

    Input features (103 dims total):
        product_id   -> Embedding(12K+1, 64)           = 64
        category     -> Embedding(27, 16)               = 16
        brand        -> Embedding(321, 16)              = 16
        price        -> normalized float                = 1
        is_store_brand -> binary                        = 1
        popularity_score -> float                       = 1
        margin_pct   -> float                           = 1
        coupon_clip_rate -> float                       = 1
        coupon_redemption_rate -> float                 = 1
        organic_purchase_ratio -> float                 = 1
    """

    def __init__(self, num_products: int = 12_001, num_categories: int = 27,
                 num_brands: int = 321, prod_embed_dim: int = 64,
                 cat_embed_dim: int = 16, brand_embed_dim: int = 16,
                 hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.product_embed = nn.Embedding(num_products, prod_embed_dim)
        self.category_embed = nn.Embedding(num_categories, cat_embed_dim)
        self.brand_embed = nn.Embedding(num_brands, brand_embed_dim)
        # 64 + 16 + 16 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 103
        input_dim = prod_embed_dim + cat_embed_dim + brand_embed_dim + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, product_id, category_id, brand_id, price,
                is_store_brand, popularity, margin_pct,
                coupon_clip_rate, coupon_redemption_rate,
                organic_purchase_ratio):
        pe = self.product_embed(product_id)         # (B, 64)
        ce = self.category_embed(category_id)       # (B, 16)
        be = self.brand_embed(brand_id)             # (B, 16)
        x = torch.cat([
            pe, ce, be,
            price.unsqueeze(1),
            is_store_brand.unsqueeze(1),
            popularity.unsqueeze(1),
            margin_pct.unsqueeze(1),
            coupon_clip_rate.unsqueeze(1),
            coupon_redemption_rate.unsqueeze(1),
            organic_purchase_ratio.unsqueeze(1),
        ], dim=1)                                   # (B, 103)
        x = F.relu(self.fc1(x))                     # (B, 256)
        x = self.fc2(x)                             # (B, 256)
        x = F.normalize(x, p=2, dim=1)              # L2 normalize
        return x


class TwoTowerModel(nn.Module):
    """Combines customer and product towers.

    Forward returns (positive_scores, negative_scores).
    Loss: BCE with logits, optionally weighted by product margin.
    """

    def __init__(self, customer_tower: CustomerTower, product_tower: ProductTower):
        super().__init__()
        self.customer_tower = customer_tower
        self.product_tower = product_tower

    def forward(self, customer_feats: dict, pos_product_feats: dict,
                neg_product_feats_list: list[dict]):
        cust_emb = self.customer_tower(**customer_feats)          # (B, 256)
        pos_emb = self.product_tower(**pos_product_feats)         # (B, 256)
        pos_scores = (cust_emb * pos_emb).sum(dim=1)             # (B,)

        neg_scores_list = []
        for neg_feats in neg_product_feats_list:
            neg_emb = self.product_tower(**neg_feats)             # (B, 256)
            neg_scores_list.append((cust_emb * neg_emb).sum(dim=1))
        neg_scores = torch.stack(neg_scores_list, dim=1)         # (B, neg_samples)

        return pos_scores, neg_scores

    @staticmethod
    def compute_loss(pos_scores, neg_scores, pos_margin=None):
        """BCE with logits, optionally margin-weighted."""
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores), reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores), reduction="none").mean(dim=1)

        if pos_margin is not None:
            # Normalize margin to [0.5, 1.5] range so all products train
            # but high-margin products get stronger gradients
            weight = 0.5 + pos_margin.clamp(0, 0.75) / 0.75
            loss = (pos_loss * weight + neg_loss * weight).mean()
        else:
            loss = (pos_loss + neg_loss).mean()
        return loss
