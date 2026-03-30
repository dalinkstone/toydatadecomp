"""Train the two-tower coupon response model.

Loads feature tables from DuckDB, constructs labeled training examples from
coupon clips, discount purchases, and organic purchases, and trains with
margin-weighted BCE loss.

Uses DuckDB for feature materialization, numpy arrays for fast lookups,
and PyTorch for GPU-accelerated training on MPS (Apple Silicon).
"""

import os
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()


class CouponResponseDataset(Dataset):
    """PyTorch dataset over labeled (customer, product, discount, label, weight) examples.

    For each example, generates neg_samples random negative products.
    Coupon non-redeemed examples (label=0) still get random negatives
    so the model learns what the customer wouldn't buy at any discount.
    """

    def __init__(self, customer_ids: np.ndarray, product_ids: np.ndarray,
                 discount_offers: np.ndarray, labels: np.ndarray,
                 weights: np.ndarray,
                 customer_features: dict[str, np.ndarray],
                 product_lookup: dict[int, dict],
                 elasticity_lookup: dict[int, dict],
                 tier_vocab: dict[str, int],
                 brand_vocab: dict[str, int],
                 category_vocab: dict[str, int],
                 norm_stats: dict[str, tuple[float, float]],
                 num_products: int, neg_samples: int = 4):
        self.customer_ids = customer_ids
        self.product_ids = product_ids
        self.discount_offers = discount_offers
        self.labels = labels
        self.weights = weights
        self.cf = customer_features
        self.pl = product_lookup
        self.el = elasticity_lookup
        self.tier_vocab = tier_vocab
        self.brand_vocab = brand_vocab
        self.category_vocab = category_vocab
        self.norm_stats = norm_stats
        self.num_products = num_products
        self.neg_samples = neg_samples
        self.product_id_list = np.array(list(product_lookup.keys()), dtype=np.int64)

    def __len__(self):
        return len(self.customer_ids)

    def _norm(self, val: float, key: str) -> float:
        mean, std = self.norm_stats.get(key, (0.0, 1.0))
        return (val - mean) / std

    def _customer_feats(self, cid: int) -> dict:
        gender_idx = int(self.cf["gender"][cid])
        gender_oh = [0.0, 0.0, 0.0]
        gender_oh[gender_idx] = 1.0
        return {
            "customer_id": cid,
            "age": self._norm(float(self.cf["age"][cid]), "age"),
            "gender_onehot": gender_oh,
            "state_id": int(self.cf["state"][cid]),
            "is_student": float(self.cf["is_student"][cid]),
            "total_spend": self._norm(float(self.cf["total_spend"][cid]), "total_spend"),
            "coupon_engagement": float(self.cf["coupon_engagement_score"][cid]),
            "coupon_redemption_rate": float(self.cf["coupon_redemption_rate"][cid]),
            "avg_basket_size": self._norm(float(self.cf["avg_basket_size"][cid]), "avg_basket_size"),
            "price_sensitivity_bucket": int(self.cf["price_sensitivity_bucket"][cid]),
        }

    def _product_feats(self, pid: int, discount_offer: float = 0.0) -> dict:
        p = self.pl.get(pid)
        e = self.el.get(pid, {})
        if p is None:
            return {
                "product_id": pid, "category_id": 0, "brand_id": 0,
                "price": 0.0, "is_store_brand": 0.0, "popularity": 0.0,
                "margin_pct": 0.0, "coupon_clip_rate": 0.0,
                "coupon_redemption_rate": 0.0, "organic_purchase_ratio": 1.0,
                "tier_id": 0, "elasticity_beta": 0.0,
                "optimal_discount": 0.0, "discount_offer": discount_offer,
            }
        return {
            "product_id": pid,
            "category_id": self.category_vocab.get(p.get("category", ""), 0),
            "brand_id": self.brand_vocab.get(p.get("brand", ""), 0),
            "price": self._norm(float(p.get("price", 0)), "price"),
            "is_store_brand": float(p.get("is_store_brand", False)),
            "popularity": float(p.get("popularity_score", 0)),
            "margin_pct": float(p.get("margin_pct", 0)),
            "coupon_clip_rate": float(p.get("coupon_clip_rate", 0)),
            "coupon_redemption_rate": float(p.get("coupon_redemption_rate", 0)),
            "organic_purchase_ratio": float(p.get("organic_purchase_ratio", 1)),
            "tier_id": self.tier_vocab.get(p.get("tier", ""), 0),
            "elasticity_beta": float(e.get("elasticity_beta", 0)),
            "optimal_discount": float(e.get("optimal_discount", 0)),
            "discount_offer": float(discount_offer),
        }

    def __getitem__(self, idx):
        cid = int(self.customer_ids[idx])
        pid = int(self.product_ids[idx])
        discount = float(self.discount_offers[idx])
        label = float(self.labels[idx])
        weight = float(self.weights[idx])

        cust = self._customer_feats(cid)
        pos = self._product_feats(pid, discount)

        # Random negatives: organic examples (discount=0) get zero-discount negatives,
        # others get random discount levels
        neg_indices = np.random.randint(len(self.product_id_list), size=self.neg_samples)
        if discount == 0.0:
            neg_discounts = np.zeros(self.neg_samples, dtype=np.float32)
        else:
            neg_discounts = np.random.uniform(0, 0.30, size=self.neg_samples).astype(np.float32)

        negs = [self._product_feats(int(self.product_id_list[i]), float(neg_discounts[i]))
                for i in range(self.neg_samples)]

        return cust, pos, negs, label, weight


def collate_fn(batch):
    """Custom collation: batch dicts of scalars into dicts of tensors."""
    custs, poss, negs_list, labels, weights = zip(*batch)
    B = len(custs)
    neg_samples = len(negs_list[0])

    def stack_customer(items):
        return {
            "customer_id": torch.tensor([x["customer_id"] for x in items], dtype=torch.long),
            "age": torch.tensor([x["age"] for x in items], dtype=torch.float32),
            "gender_onehot": torch.tensor([x["gender_onehot"] for x in items], dtype=torch.float32),
            "state_id": torch.tensor([x["state_id"] for x in items], dtype=torch.long),
            "is_student": torch.tensor([x["is_student"] for x in items], dtype=torch.float32),
            "total_spend": torch.tensor([x["total_spend"] for x in items], dtype=torch.float32),
            "coupon_engagement": torch.tensor([x["coupon_engagement"] for x in items], dtype=torch.float32),
            "coupon_redemption_rate": torch.tensor([x["coupon_redemption_rate"] for x in items], dtype=torch.float32),
            "avg_basket_size": torch.tensor([x["avg_basket_size"] for x in items], dtype=torch.float32),
            "price_sensitivity_bucket": torch.tensor([x["price_sensitivity_bucket"] for x in items], dtype=torch.long),
        }

    def stack_product(items):
        return {
            "product_id": torch.tensor([x["product_id"] for x in items], dtype=torch.long),
            "category_id": torch.tensor([x["category_id"] for x in items], dtype=torch.long),
            "brand_id": torch.tensor([x["brand_id"] for x in items], dtype=torch.long),
            "price": torch.tensor([x["price"] for x in items], dtype=torch.float32),
            "is_store_brand": torch.tensor([x["is_store_brand"] for x in items], dtype=torch.float32),
            "popularity": torch.tensor([x["popularity"] for x in items], dtype=torch.float32),
            "margin_pct": torch.tensor([x["margin_pct"] for x in items], dtype=torch.float32),
            "coupon_clip_rate": torch.tensor([x["coupon_clip_rate"] for x in items], dtype=torch.float32),
            "coupon_redemption_rate": torch.tensor([x["coupon_redemption_rate"] for x in items], dtype=torch.float32),
            "organic_purchase_ratio": torch.tensor([x["organic_purchase_ratio"] for x in items], dtype=torch.float32),
            "tier_id": torch.tensor([x["tier_id"] for x in items], dtype=torch.long),
            "elasticity_beta": torch.tensor([x["elasticity_beta"] for x in items], dtype=torch.float32),
            "optimal_discount": torch.tensor([x["optimal_discount"] for x in items], dtype=torch.float32),
            "discount_offer": torch.tensor([x["discount_offer"] for x in items], dtype=torch.float32),
        }

    cust_batch = stack_customer(custs)
    pos_batch = stack_product(poss)

    neg_batches = []
    for ni in range(neg_samples):
        neg_items = [negs_list[b][ni] for b in range(B)]
        neg_batches.append(stack_product(neg_items))

    pos_margin = pos_batch["margin_pct"].clone()
    labels_t = torch.tensor(labels, dtype=torch.float32)
    weights_t = torch.tensor(weights, dtype=torch.float32)

    return cust_batch, pos_batch, neg_batches, labels_t, weights_t, pos_margin


def select_device(preference: str) -> torch.device:
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        console.print("[yellow]MPS not available, falling back to CPU[/yellow]")
        return torch.device("cpu")
    if preference == "auto":
        return torch.device("cpu")
    return torch.device(preference)


def move_to(d: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in d.items()}


def extract_embeddings(model, customer_features, product_lookup, elasticity_lookup,
                       tier_vocab, brand_vocab, category_vocab, norm_stats,
                       device, output_dir):
    """Extract all customer and product embeddings post-training."""
    console.print("\n[bold]Extracting embeddings...[/bold]")
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Product embeddings: all at once (with discount_offer=0.0 for base embeddings)
    console.print("  Product embeddings...")
    product_ids = sorted(product_lookup.keys())
    num_products = len(product_ids)

    dummy_ds = CouponResponseDataset(
        np.array([1]), np.array([1]), np.array([0.0]),
        np.array([1.0]), np.array([1.0]),
        customer_features, product_lookup, elasticity_lookup, tier_vocab,
        brand_vocab, category_vocab, norm_stats,
        num_products=num_products, neg_samples=0)

    prod_feats_list = [dummy_ds._product_feats(pid, 0.0) for pid in product_ids]

    def stack_product_list(items):
        return {
            "product_id": torch.tensor([x["product_id"] for x in items], dtype=torch.long),
            "category_id": torch.tensor([x["category_id"] for x in items], dtype=torch.long),
            "brand_id": torch.tensor([x["brand_id"] for x in items], dtype=torch.long),
            "price": torch.tensor([x["price"] for x in items], dtype=torch.float32),
            "is_store_brand": torch.tensor([x["is_store_brand"] for x in items], dtype=torch.float32),
            "popularity": torch.tensor([x["popularity"] for x in items], dtype=torch.float32),
            "margin_pct": torch.tensor([x["margin_pct"] for x in items], dtype=torch.float32),
            "coupon_clip_rate": torch.tensor([x["coupon_clip_rate"] for x in items], dtype=torch.float32),
            "coupon_redemption_rate": torch.tensor([x["coupon_redemption_rate"] for x in items], dtype=torch.float32),
            "organic_purchase_ratio": torch.tensor([x["organic_purchase_ratio"] for x in items], dtype=torch.float32),
            "tier_id": torch.tensor([x["tier_id"] for x in items], dtype=torch.long),
            "elasticity_beta": torch.tensor([x["elasticity_beta"] for x in items], dtype=torch.float32),
            "optimal_discount": torch.tensor([x["optimal_discount"] for x in items], dtype=torch.float32),
            "discount_offer": torch.tensor([x["discount_offer"] for x in items], dtype=torch.float32),
        }

    with torch.inference_mode():
        prod_batch = move_to(stack_product_list(prod_feats_list), device)
        prod_embeddings = model.product_tower(**prod_batch).cpu().numpy()

    prod_path = os.path.join(output_dir, "product_embeddings.npy")
    np.save(prod_path, prod_embeddings)
    console.print(f"  Saved product_embeddings: {prod_embeddings.shape}")

    np.save(os.path.join(output_dir, "product_ids.npy"),
            np.array(product_ids, dtype=np.int64))

    # Customer embeddings: in chunks (vectorized)
    console.print("  Customer embeddings (chunked, vectorized)...")
    cf = customer_features
    max_cid = cf["age"].shape[0]
    chunk_size = 100_000
    cust_embeddings = np.zeros((max_cid, 256), dtype=np.float32)

    def _vnorm(arr, key):
        mean, std = norm_stats.get(key, (0.0, 1.0))
        return ((arr - mean) / std).astype(np.float32)

    with torch.inference_mode():
        for start in range(1, max_cid, chunk_size):
            end = min(start + chunk_size, max_cid)
            n = end - start

            gender_idx = cf["gender"][start:end].astype(np.int64)
            gender_oh = np.zeros((n, 3), dtype=np.float32)
            gender_oh[np.arange(n), np.clip(gender_idx, 0, 2)] = 1.0

            batch = {
                "customer_id": torch.arange(start, end, dtype=torch.long, device=device),
                "age": torch.from_numpy(_vnorm(cf["age"][start:end], "age")).to(device),
                "gender_onehot": torch.from_numpy(gender_oh).to(device),
                "state_id": torch.from_numpy(cf["state"][start:end].astype(np.int64)).to(device),
                "is_student": torch.from_numpy(cf["is_student"][start:end].copy()).to(device),
                "total_spend": torch.from_numpy(_vnorm(cf["total_spend"][start:end], "total_spend")).to(device),
                "coupon_engagement": torch.from_numpy(cf["coupon_engagement_score"][start:end].copy()).to(device),
                "coupon_redemption_rate": torch.from_numpy(cf["coupon_redemption_rate"][start:end].copy()).to(device),
                "avg_basket_size": torch.from_numpy(_vnorm(cf["avg_basket_size"][start:end], "avg_basket_size")).to(device),
                "price_sensitivity_bucket": torch.from_numpy(cf["price_sensitivity_bucket"][start:end].astype(np.int64)).to(device),
            }
            chunk_emb = model.customer_tower(**batch).cpu().numpy()
            cust_embeddings[start:end] = chunk_emb

            if (start // chunk_size) % 10 == 0:
                console.print(f"    {start:,} / {max_cid - 1:,}")

    cust_path = os.path.join(output_dir, "customer_embeddings.npy")
    np.save(cust_path, cust_embeddings)
    console.print(f"  Saved customer_embeddings: {cust_embeddings.shape}")


def compute_discount_curves(model, product_lookup, elasticity_lookup, tier_vocab,
                            brand_vocab, category_vocab, norm_stats,
                            customer_features, device, output_dir):
    """Compute discount response curves: average score at each discount level per product.

    For each product, runs inference at discount levels [0%, 5%, 10%, 15%, 20%, 25%, 30%]
    averaged across a sample of 10,000 customers.

    Saves to data/model/discount_response_curves.parquet.
    """
    console.print("\n[bold]Computing discount response curves...[/bold]")
    model.eval()

    discount_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    product_ids = sorted(product_lookup.keys())

    # Sample 10K customers for scoring
    cf = customer_features
    max_cid = cf["age"].shape[0]
    sample_size = min(10_000, max_cid - 1)
    sample_cids = np.random.choice(np.arange(1, max_cid), size=sample_size, replace=False)

    dummy_ds = CouponResponseDataset(
        np.array([1]), np.array([1]), np.array([0.0]),
        np.array([1.0]), np.array([1.0]),
        customer_features, product_lookup, elasticity_lookup, tier_vocab,
        brand_vocab, category_vocab, norm_stats,
        num_products=len(product_ids), neg_samples=0)

    def _vnorm(arr, key):
        mean, std = norm_stats.get(key, (0.0, 1.0))
        return ((arr - mean) / std).astype(np.float32)

    with torch.inference_mode():
        # Compute customer embeddings for sample
        n = len(sample_cids)
        gender_idx = cf["gender"][sample_cids].astype(np.int64)
        gender_oh = np.zeros((n, 3), dtype=np.float32)
        gender_oh[np.arange(n), np.clip(gender_idx, 0, 2)] = 1.0

        cust_batch = {
            "customer_id": torch.from_numpy(sample_cids.astype(np.int64)).to(device),
            "age": torch.from_numpy(_vnorm(cf["age"][sample_cids], "age")).to(device),
            "gender_onehot": torch.from_numpy(gender_oh).to(device),
            "state_id": torch.from_numpy(cf["state"][sample_cids].astype(np.int64)).to(device),
            "is_student": torch.from_numpy(cf["is_student"][sample_cids].copy()).to(device),
            "total_spend": torch.from_numpy(_vnorm(cf["total_spend"][sample_cids], "total_spend")).to(device),
            "coupon_engagement": torch.from_numpy(cf["coupon_engagement_score"][sample_cids].copy()).to(device),
            "coupon_redemption_rate": torch.from_numpy(cf["coupon_redemption_rate"][sample_cids].copy()).to(device),
            "avg_basket_size": torch.from_numpy(_vnorm(cf["avg_basket_size"][sample_cids], "avg_basket_size")).to(device),
            "price_sensitivity_bucket": torch.from_numpy(cf["price_sensitivity_bucket"][sample_cids].astype(np.int64)).to(device),
        }
        cust_embs = model.customer_tower(**cust_batch)  # (10K, 256)

        def stack_product_list(items):
            return {
                "product_id": torch.tensor([x["product_id"] for x in items], dtype=torch.long),
                "category_id": torch.tensor([x["category_id"] for x in items], dtype=torch.long),
                "brand_id": torch.tensor([x["brand_id"] for x in items], dtype=torch.long),
                "price": torch.tensor([x["price"] for x in items], dtype=torch.float32),
                "is_store_brand": torch.tensor([x["is_store_brand"] for x in items], dtype=torch.float32),
                "popularity": torch.tensor([x["popularity"] for x in items], dtype=torch.float32),
                "margin_pct": torch.tensor([x["margin_pct"] for x in items], dtype=torch.float32),
                "coupon_clip_rate": torch.tensor([x["coupon_clip_rate"] for x in items], dtype=torch.float32),
                "coupon_redemption_rate": torch.tensor([x["coupon_redemption_rate"] for x in items], dtype=torch.float32),
                "organic_purchase_ratio": torch.tensor([x["organic_purchase_ratio"] for x in items], dtype=torch.float32),
                "tier_id": torch.tensor([x["tier_id"] for x in items], dtype=torch.long),
                "elasticity_beta": torch.tensor([x["elasticity_beta"] for x in items], dtype=torch.float32),
                "optimal_discount": torch.tensor([x["optimal_discount"] for x in items], dtype=torch.float32),
                "discount_offer": torch.tensor([x["discount_offer"] for x in items], dtype=torch.float32),
            }

        results = {"product_id": product_ids}

        for disc_level in discount_levels:
            prod_feats = [dummy_ds._product_feats(pid, disc_level) for pid in product_ids]
            prod_batch = move_to(stack_product_list(prod_feats), device)
            prod_embs = model.product_tower(**prod_batch)  # (num_products, 256)

            # Average score across sample customers: (10K, num_products) -> (num_products,)
            scores = torch.mm(cust_embs, prod_embs.t())
            avg_scores = scores.mean(dim=0).cpu().numpy()

            col_name = f"score_at_{int(disc_level * 100)}pct"
            results[col_name] = avg_scores
            console.print(f"  Discount {int(disc_level * 100)}%: "
                          f"avg score {avg_scores.mean():.4f}")

    df = pd.DataFrame(results)
    path = os.path.join(output_dir, "discount_response_curves.parquet")
    df.to_parquet(path, index=False)
    console.print(f"  Saved: {path} ({len(df)} products x {len(discount_levels)} levels)")


@click.command()
@click.option("--db-path", default="data/db/cvs_analytics.duckdb",
              help="DuckDB database path.")
@click.option("--epochs", default=5, help="Number of training epochs.")
@click.option("--batch-size", default=8192, help="Training batch size.")
@click.option("--lr", default=1e-3, type=float, help="Learning rate.")
@click.option("--sample-pct", default=1.0, type=float,
              help="Percent of transactions to sample (1.0 = 1%%).")
@click.option("--device", default="auto",
              type=click.Choice(["auto", "mps", "cpu"]))
@click.option("--output-dir", default="data/model/",
              help="Model checkpoint and embedding output directory.")
@click.option("--neg-samples", default=4, help="Negative samples per positive.")
@click.option("--margin-weight/--no-margin-weight", default=True,
              help="Use margin-weighted loss.")
@click.option("--skip-features", is_flag=True,
              help="Skip feature engineering (assumes tables exist).")
def main(db_path: str, epochs: int, batch_size: int, lr: float,
         sample_pct: float, device: str, output_dir: str,
         neg_samples: int, margin_weight: bool, skip_features: bool):
    """Train the two-tower coupon response model."""
    from ml.features import FeatureStore
    from ml.two_tower import CustomerTower, ProductTower, TwoTowerModel

    console.print("[bold]Two-Tower Coupon Response Model Training[/bold]")
    console.print(f"  DB: {db_path}")
    console.print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    console.print(f"  Sample: {sample_pct}%, Neg samples: {neg_samples}")
    console.print(f"  Margin-weighted loss: {margin_weight}")

    os.makedirs(output_dir, exist_ok=True)
    dev = select_device(device)
    console.print(f"  Device: {dev}\n")

    # -- Feature preparation --
    fs = FeatureStore(db_path)
    if not skip_features or not fs.has_features():
        fs.build_product_features()
        fs.build_customer_features()
        fs.build_product_tiers()
        fs.build_coupon_response_pairs(sample_pct)
    else:
        console.print("[yellow]Skipping feature engineering (tables exist).[/yellow]")

    customer_features = fs.export_customer_lookup()
    product_lookup = fs.export_product_lookup()
    elasticity_lookup = fs.export_elasticity_lookup()
    tier_vocab = fs.export_tier_vocab()
    brand_vocab = fs.export_brand_vocab()
    category_vocab = fs.export_category_vocab()
    norm_stats = fs.export_normalization_stats()
    train_cids, train_pids, train_discounts, train_labels, train_weights = \
        fs.export_coupon_training_data()

    num_products = len(product_lookup)
    num_brands = len(brand_vocab) + 1
    num_categories = len(category_vocab) + 1
    num_states = len(fs.export_state_vocab()) + 1
    num_tiers = max(tier_vocab.values()) + 1
    max_customer_id = int(customer_features["age"].shape[0])

    console.print(f"\n[bold]Model dimensions:[/bold]")
    console.print(f"  Customers: {max_customer_id - 1:,}, Products: {num_products:,}")
    console.print(f"  Brands: {num_brands}, Categories: {num_categories}, "
                  f"States: {num_states}, Tiers: {num_tiers}")
    console.print(f"  Training examples: {len(train_cids):,}")
    console.print(f"    Labels: {(train_labels == 1).sum():,} positive, "
                  f"{(train_labels == 0).sum():,} negative\n")

    fs.close()

    # -- Model --
    customer_tower = CustomerTower(
        num_customers=max_customer_id,
        num_states=num_states)
    product_tower = ProductTower(
        num_products=num_products + 1,
        num_categories=num_categories,
        num_brands=num_brands,
        num_tiers=num_tiers)
    model = TwoTowerModel(customer_tower, product_tower).to(dev)

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Model parameters: {total_params:,} ({total_params * 4 / 1e9:.2f} GB)[/bold]\n")

    # -- Dataset & DataLoader --
    dataset = CouponResponseDataset(
        train_cids, train_pids, train_discounts, train_labels, train_weights,
        customer_features, product_lookup, elasticity_lookup, tier_vocab,
        brand_vocab, category_vocab, norm_stats,
        num_products=num_products, neg_samples=neg_samples)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
        pin_memory=False, drop_last=True,
        persistent_workers=True)

    # -- Optimizer & Scheduler --
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # -- Training loop --
    console.print("[bold]Training...[/bold]\n")
    t0_total = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0_epoch = time.time()
        num_batches = len(loader)

        with Progress(
            TextColumn(f"Epoch {epoch + 1}/{epochs}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=num_batches)

            for cust_batch, pos_batch, neg_batches, labels, weights, pos_margin in loader:
                cust_batch = move_to(cust_batch, dev)
                pos_batch = move_to(pos_batch, dev)
                neg_batches = [move_to(nb, dev) for nb in neg_batches]
                labels = labels.to(dev)
                weights = weights.to(dev)
                pos_margin = pos_margin.to(dev) if margin_weight else None

                optimizer.zero_grad(set_to_none=True)
                pos_scores, neg_scores = model(cust_batch, pos_batch, neg_batches)
                loss = TwoTowerModel.compute_loss(
                    pos_scores, neg_scores, labels, weights, pos_margin)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                progress.advance(task)

        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - t0_epoch
        current_lr = scheduler.get_last_lr()[0]
        console.print(f"  Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s\n")

        # Checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "norm_stats": norm_stats,
            "brand_vocab": brand_vocab,
            "category_vocab": category_vocab,
            "tier_vocab": tier_vocab,
        }, ckpt_path)
        console.print(f"  Checkpoint: {ckpt_path}")

    total_time = time.time() - t0_total
    console.print(f"\n[bold green]Training complete[/bold green] ({total_time:.1f}s)")

    # -- Extract embeddings --
    extract_embeddings(model, customer_features, product_lookup, elasticity_lookup,
                       tier_vocab, brand_vocab, category_vocab, norm_stats,
                       dev, output_dir)

    # -- Compute discount response curves --
    compute_discount_curves(model, product_lookup, elasticity_lookup, tier_vocab,
                            brand_vocab, category_vocab, norm_stats,
                            customer_features, dev, output_dir)

    console.print(f"\n[bold green]All done![/bold green]")
    console.print(f"  Model: {output_dir}")
    console.print(f"  Embeddings: customer_embeddings.npy, product_embeddings.npy")
    console.print(f"  Discount curves: discount_response_curves.parquet")


if __name__ == "__main__":
    main()
