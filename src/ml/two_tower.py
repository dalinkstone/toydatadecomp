"""Two-tower (dual encoder) coupon response model.

CustomerTower encodes customer features into a 256-dim embedding.
ProductTower encodes product features (with discount context) into a 256-dim embedding.
Coupon response score = dot product of normalized embeddings.

Enhanced with coupon engagement features, product tiers, elasticity data,
and discount_offer input to predict customer response to discount offers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomerTower(nn.Module):
    """Encodes customer features into a 256-dim L2-normalized embedding.

    Input features (97 dims total):
        customer_id  -> Embedding(10M+1, 64)           = 64
        age          -> normalized float                = 1
        gender       -> one-hot (F, M, NB)              = 3
        state        -> Embedding(51, 16)               = 16
        is_student   -> binary                          = 1
        total_spend  -> normalized float                = 1
        coupon_engagement_score -> float                = 1
        coupon_redemption_rate  -> float                = 1
        avg_basket_size -> normalized float             = 1
        price_sensitivity_bucket -> Embedding(5, 8)    = 8
    """

    def __init__(self, num_customers: int = 10_000_001, num_states: int = 51,
                 cust_embed_dim: int = 64, state_embed_dim: int = 16,
                 num_price_buckets: int = 5, price_bucket_dim: int = 8,
                 hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.customer_embed = nn.Embedding(num_customers, cust_embed_dim)
        self.state_embed = nn.Embedding(num_states, state_embed_dim)
        self.price_sensitivity_embed = nn.Embedding(num_price_buckets, price_bucket_dim)
        # 64 + 1 + 3 + 16 + 1 + 1 + 1 + 1 + 1 + 8 = 97
        input_dim = cust_embed_dim + 1 + 3 + state_embed_dim + 1 + 1 + 1 + 1 + 1 + price_bucket_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, customer_id, age, gender_onehot, state_id,
                is_student, total_spend, coupon_engagement,
                coupon_redemption_rate, avg_basket_size,
                price_sensitivity_bucket):
        ce = self.customer_embed(customer_id)       # (B, 64)
        se = self.state_embed(state_id)             # (B, 16)
        pse = self.price_sensitivity_embed(price_sensitivity_bucket)  # (B, 8)
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
            pse,
        ], dim=1)                                   # (B, 97)
        x = F.relu(self.fc1(x))                     # (B, 256)
        x = self.fc2(x)                             # (B, 256)
        x = F.normalize(x, p=2, dim=1)              # L2 normalize
        return x


class ProductTower(nn.Module):
    """Encodes product features into a 256-dim L2-normalized embedding.

    Input features (114 dims total):
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
        tier         -> Embedding(6, 8)                 = 8
        elasticity_beta -> float                        = 1
        optimal_discount -> float                       = 1
        discount_offer -> float (0.0-0.50)              = 1
    """

    def __init__(self, num_products: int = 12_001, num_categories: int = 27,
                 num_brands: int = 321, prod_embed_dim: int = 64,
                 cat_embed_dim: int = 16, brand_embed_dim: int = 16,
                 num_tiers: int = 6, tier_embed_dim: int = 8,
                 hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.product_embed = nn.Embedding(num_products, prod_embed_dim)
        self.category_embed = nn.Embedding(num_categories, cat_embed_dim)
        self.brand_embed = nn.Embedding(num_brands, brand_embed_dim)
        self.tier_embed = nn.Embedding(num_tiers, tier_embed_dim)
        # 64 + 16 + 16 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 8 + 1 + 1 + 1 = 114
        input_dim = (prod_embed_dim + cat_embed_dim + brand_embed_dim +
                     1 + 1 + 1 + 1 + 1 + 1 + 1 +
                     tier_embed_dim + 1 + 1 + 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, product_id, category_id, brand_id, price,
                is_store_brand, popularity, margin_pct,
                coupon_clip_rate, coupon_redemption_rate,
                organic_purchase_ratio,
                tier_id, elasticity_beta, optimal_discount, discount_offer):
        pe = self.product_embed(product_id)         # (B, 64)
        ce = self.category_embed(category_id)       # (B, 16)
        be = self.brand_embed(brand_id)             # (B, 16)
        te = self.tier_embed(tier_id)               # (B, 8)
        x = torch.cat([
            pe, ce, be,
            price.unsqueeze(1),
            is_store_brand.unsqueeze(1),
            popularity.unsqueeze(1),
            margin_pct.unsqueeze(1),
            coupon_clip_rate.unsqueeze(1),
            coupon_redemption_rate.unsqueeze(1),
            organic_purchase_ratio.unsqueeze(1),
            te,
            elasticity_beta.unsqueeze(1),
            optimal_discount.unsqueeze(1),
            discount_offer.unsqueeze(1),
        ], dim=1)                                   # (B, 114)
        x = F.relu(self.fc1(x))                     # (B, 256)
        x = self.fc2(x)                             # (B, 256)
        x = F.normalize(x, p=2, dim=1)              # L2 normalize
        return x


class TwoTowerModel(nn.Module):
    """Combines customer and product towers.

    Forward returns (positive_scores, negative_scores).
    Loss: BCE with logits, label-aware for coupon non-redemptions,
    optionally weighted by product margin and example importance.
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
    def compute_loss(pos_scores, neg_scores, labels, weights=None, pos_margin=None):
        """BCE with logits, label-aware positive slot with optional weighting.

        Args:
            pos_scores: (B,) scores for the primary product slot.
            neg_scores: (B, neg_samples) scores for random negative products.
            labels: (B,) target for positive slot (1.0 for purchases/redeemed,
                    0.0 for coupon non-redemptions).
            weights: (B,) per-example importance weights (e.g., 0.5 for organic).
            pos_margin: (B,) product margin for margin-weighted loss.
        """
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, labels, reduction="none")
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores), reduction="none").mean(dim=1)

        combined = pos_loss + neg_loss
        if pos_margin is not None:
            margin_weight = 0.5 + pos_margin.clamp(0, 0.75) / 0.75
            combined = combined * margin_weight
        if weights is not None:
            combined = combined * weights
        return combined.mean()
