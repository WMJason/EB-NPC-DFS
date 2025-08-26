import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import shutil
from shutil import copyfile

import re
import gzip
import json
import arviz as az
from tqdm import tqdm
import pickle

filename = '0_sp_lookups.pkl'
with open(filename, 'rb') as f:
    sp_lookup = pickle.load(f)

print(f"✅ Indexed {len(sp_lookup)} valid shortest paths.")

def predict_spatial_latent_term_for_new_sites(
        new_midpoints_ids,
        knots_ids,
        psi_nc_path,
        path_to_jsonl="0_paths_with_legs_ALL_combined.jsonl.gz",
        sp_lookup=None,
        return_kernel=False,
        h=1000,
    ):

    # === Load calibrated psi (posterior mean) ===
    trace = az.from_netcdf(psi_nc_path)
    psi_post_mean = trace.posterior["psi"].mean(dim=("chain", "draw")).values  # shape: (n_knots,)

    if psi_post_mean.shape[0] != len(knots_ids):
        raise ValueError(f"psi length mismatch: got {psi_post_mean.shape[0]} values, expected {len(knots_ids)} knots")

    # === Compute reweighted kernel K_new ===
    K_weighteds = np.zeros((len(new_midpoints_ids), len(knots_ids)))
    midpoint_set = set(new_midpoints_ids)
    knot_set = set(knots_ids)

    if sp_lookup is None:
        sp_lookup = {}
        print("📦 Indexing shortest paths for prediction...")
        with gzip.open(path_to_jsonl, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Indexing JSONL lines"):
                sp_info = json.loads(line)
                i, j = sp_info["origin_id"], sp_info["destination_id"]

                if (i in midpoint_set and j in knot_set) or (j in midpoint_set and i in knot_set):
                    key = tuple(sorted((i, j)))
                    sp_lookup[key] = sp_info

        print(f"✅ Indexed {len(sp_lookup)} valid shortest paths.")

    print("🧮 Computing kernel matrix for new midpoints...")
    for i_idx, midpoint_id in enumerate(tqdm(new_midpoints_ids, desc="New midpoints")):
        for j_idx, knot_id in enumerate(knots_ids):
            if midpoint_id == knot_id:
                sp = 0
                weights = [1]
            else:
                key = tuple(sorted((midpoint_id, knot_id)))
                sp_info = sp_lookup.get(key)

                if not sp_info:
                    sp = np.inf
                    weights = [1]
                else:
                    sp = sp_info.get("distance", 0)
                    weights = []
                    total_legs = sp_info.get("total_legs", [])
                    ac_legs = sp_info.get("ac_legs", [])

                    for total, ac in zip(total_legs, ac_legs):
                        if ac > 2:
                            w = (2 / ac) * (1 / (ac - 1))
                        else:
                            w = 1
                        weights.append(w)

                    if not weights:
                        weights = [1]

            if sp == np.inf:
              K = 0
            else:
              reweight = np.prod(weights)
              K = (reweight ** 0.5) * max(1 - abs(sp / h), 0)
            K_weighteds[i_idx, j_idx] = K

    # === Predict spatial latent term ===
    Z_new = K_weighteds @ psi_post_mean

    if return_kernel:
        return Z_new, K_weighteds
    else:
        return Z_new

##################
output_folder = '6_NPC_DFS_after2-2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    try:
        for ea in os.listdir(output_folder):
            os.remove(output_folder + '/' + ea)
    except:
        for ea in os.listdir(output_folder):
            shutil.rmtree(output_folder + '/' + ea)


collision_types = ['overall', 'PDO', 'Severe']
for collision_type in collision_types:


    ####PREPARING FOR PREDICTION

    ####FOR DFS sites
    dfs_file = f'0_data_DFS_sites_{collision_type}_road_props.csv'
    nondfs_file = f'0_data_nonDFS_sites_{collision_type}_road_props.csv'

    df_dfs_segs = pd.read_csv(dfs_file)
    df = df_dfs_segs.copy()

    data_dfs = pd.DataFrame({
    'Index': df["Index"].values.tolist(),
    'length_m': df["Length"].values.tolist(),
    'years': df["before_years"].values.tolist(),
    'traffic_volume': df['before_avgADT'].values.tolist(),
    'observed_crashes': df['(10)Observed crash frequency in before period'].values.tolist(),
    'x_coord': df['Longitude'].values.tolist(),
    'y_coord': df['Latitude'].values.tolist(),
    'speed_limit': [1 if x > 40 else 0 for x in df['speed_osm'].values.tolist()],
    'lanes': [1 if x > 2 else 0 for x in df['lanes_osm'].values.tolist()],
    'if_dfs': 0,
    })

    print('len(data_dfs):', len(data_dfs))

    for col in ['traffic_volume','observed_crashes']:
      data_dfs[col] = pd.to_numeric(data_dfs[col], errors='coerce')
    data_dfs = data_dfs.dropna()
    print('len(data_dfs):', len(data_dfs))

    ####FOR nonDFS sites
    df_nondfs_segs = pd.read_csv(nondfs_file)
    df = df_nondfs_segs.copy()

    data_nondfs = pd.DataFrame({
      'Index': df["Index"].values.tolist(),
      'length_m': df["Length"].values.tolist(),
      'years': 10,
      'traffic_volume': df['avgADT'].values.tolist(),
      'observed_crashes': df['SumCollisions'].values.tolist(),
      'x_coord': df['Longitude'].values.tolist(),
      'y_coord': df['Latitude'].values.tolist(),
      'speed_limit': [1 if x > 40 else 0 for x in df['speed_osm'].values.tolist()],
      'lanes': [1 if x > 2 else 0 for x in df['lanes_osm'].values.tolist()],
      'if_dfs': 0,
    })

    print('len(data_nondfs):', len(data_nondfs))

    for col in ['traffic_volume','observed_crashes']:
      data_nondfs[col] = pd.to_numeric(data_nondfs[col], errors='coerce')
    data_nondfs = data_nondfs.dropna()
    print('len(data_dfs):', len(data_dfs))

    data = pd.concat([data_dfs, data_nondfs])
    print('len(data):', len(data))
    midpoints_ids = data["Index"].values.tolist()

    ##for DFS before-period prediction
    model_folder = '5_k1066_h1000_step2-2'
    traffic_volume_log = np.log(data['traffic_volume'].values)
    traffic_volume_log_mean = traffic_volume_log.mean()
    traffic_volume_log_std = traffic_volume_log.std()

    # Construct X_new -after peroid
    df_dfs_segs = pd.read_csv(dfs_file)
    df = df_dfs_segs.copy()

    data_dfs = pd.DataFrame({
    'Index': df["Index"].values.tolist(),
    'length_m': df["Length"].values.tolist(),
    'years': df["after_years"].values.tolist(),
    'traffic_volume': df['after_avgADT'].values.tolist(),
    'observed_crashes': df['(13)Observed crash frequency in after period'].values.tolist(),
    'x_coord': df['Longitude'].values.tolist(),
    'y_coord': df['Latitude'].values.tolist(),
    'speed_limit': [1 if x > 40 else 0 for x in df['speed_osm'].values.tolist()],
    'lanes': [1 if x > 2 else 0 for x in df['lanes_osm'].values.tolist()],
    'if_dfs': 0,
    })

    print('len(data_dfs):', len(data_dfs))

    for col in ['traffic_volume','observed_crashes']:
      data_dfs[col] = pd.to_numeric(data_dfs[col], errors='coerce')
    data_dfs = data_dfs.dropna()
    print('len(data_dfs):', len(data_dfs))

    traffic_volume_log = np.log(data_dfs['traffic_volume'].values)
    traffic_volume_log = (traffic_volume_log - traffic_volume_log_mean) / traffic_volume_log_std
    data_dfs['log_traffic_volume_std'] = traffic_volume_log
    X_new = data_dfs[['log_traffic_volume_std']].values  # shape (n_beta,)

    trace_file = model_folder + f'/5_trace_k1066_h1000_{collision_type}_Triangular.nc'

    new_ids = data_dfs['Index'].values.tolist()
    knots_ids_path = model_folder + '/5_knots_ids_k1066.pkl'
    with open(knots_ids_path, 'rb') as f:
        knots_ids = pickle.load(f)

    # Extract posterior draws means
    trace = az.from_netcdf(trace_file)
    beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
    beta0_mean = trace.posterior["beta0"].mean(dim=("chain", "draw")).values

    psi_nc_path = trace_file
    Z_new, K_weighteds = predict_spatial_latent_term_for_new_sites(
        new_midpoints_ids=new_ids,
        knots_ids=knots_ids,
        psi_nc_path=psi_nc_path,
        path_to_jsonl="0_paths_with_legs_ALL_combined.jsonl.gz",
        sp_lookup=sp_lookup,
        return_kernel=True
    )

    print('collision type:',collision_type,'step 2-2, all sites non-treatement predicting DFS-before == beta_mean:',beta_mean,'beta0_mean:',beta0_mean)
    print('')


    with pm.Model() as model:

        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=X_new.shape[1])
        var_psi = pm.Gamma("var_psi", alpha=2, beta=2)  # variance
        sigma_psi = pm.Deterministic("sigma_psi", pm.math.sqrt(var_psi))
        psi = pm.Normal("psi", mu=0, sigma=sigma_psi, shape=len(knots_ids))

        # Spatial random effects with reweighted kernel
        Z = pm.math.dot(K_weighteds, psi)

        # Expected crashes (log-link with offset)
        log_lambda = pm.math.log(data_dfs['length_m'].values) + pm.math.log(data_dfs['years'].values) + beta0 + pm.math.dot(X_new, beta) + Z
        lambda_ = pm.Deterministic("lambda_", pm.math.exp(log_lambda))

    with model:
        posterior_pred = pm.sample_posterior_predictive(
            trace,
            var_names=["lambda_"],  # or just the outputs you want
            random_seed=42
        )

    # Suppose this is your lambda samples
    lambda_samples = posterior_pred.posterior_predictive["lambda_"].values  # shape (2, 3000, 86)

    # Step 1: Combine chains and draws
    combined_samples = lambda_samples.reshape(-1, lambda_samples.shape[-1])  # shape (6000, 86)

    # Step 2: Compute mean for each row
    mean_vals = np.mean(combined_samples, axis=0)  # shape (86,)
    std_vals = np.std(combined_samples, axis=0)  # shape (86,)

    # Step 3: Compute 95% HDI using ArviZ
    hdi_bounds = az.hdi(combined_samples, hdi_prob=0.95)  # shape (86, 2), columns are [lower, upper]

    lower_bounds = hdi_bounds[:, 0]
    upper_bounds = hdi_bounds[:, 1]


    df_dfs_segs['predicted_after2-2_mcmc'] = mean_vals
    df_dfs_segs['predicted_after2-2_mcmc_std'] = std_vals

    lower_upper = az.hdi(trace, var_names=["lambda_"], hdi_prob=0.95)
    df_dfs_segs['predicted_after2-2_mcmc_2.5%'] = lower_bounds
    df_dfs_segs['predicted_after2-2_mcmc_97.5%'] = upper_bounds
    df_dfs_segs.to_csv(output_folder+ f'/6_DFS_sites_pred_after2-2_{collision_type}.csv')