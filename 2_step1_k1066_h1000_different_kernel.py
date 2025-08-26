import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist
import os
import shutil
from shutil import copyfile
import gzip
import json
from tqdm import tqdm
import pickle


collision_types = ['overall','PDO','Severe']
kernel_functions = ['Triangular','Epanechnikov','Quartic']

h=1000
k=1066

output_folder = '2_modelling_results_nospeed_lanes-k1066_h1000_diffferent_kernel'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    try:
        for ea in os.listdir(output_folder):
            os.remove(output_folder + '/' + ea)
    except:
        for ea in os.listdir(output_folder):
            shutil.rmtree(output_folder + '/' + ea)

for collision_type in collision_types:

  dfs_file = '0_data_DFS_sites_'+collision_type+'_road_props.csv'
  nondfs_file = '0_data_nonDFS_sites_'+collision_type+'_road_props.csv'
  ##################

  ####FOR DFS sites
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

  try:
    knots_df = pd.read_csv('0_knots_k'+str(k)+'_coords.csv')
    knots_ids = knots_df['medoid_id'].values.tolist()
  except:
    knots_df = []

    # === Compute reweighted kernel matrix ===

    # Build set of used midpoint IDs from the large file
    used_ids = set()

    with gzip.open("0_paths_with_legs_ALL_combined.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning for knots"):
            record = json.loads(line)
            o_id = record["origin_id"]
            d_id = record["destination_id"]
            if o_id in midpoints_ids:
                used_ids.add(o_id)
            if d_id in midpoints_ids:
                used_ids.add(d_id)

    # Now define knots_ids
    knots_ids = sorted(list(used_ids))


  filename = output_folder+'/2_knots_ids_k'+str(k)+'.pkl'
  with open(filename, 'wb') as f:
      pickle.dump(knots_ids, f)


  print(f"✅ Found {len(knots_ids)} knots out of {len(midpoints_ids)} midpoints.")


  def compute_reweighted_kernels(midpoints_ids, knots_ids, kernel_function='Triangular', path_to_jsonl="0_paths_with_legs_ALL_combined.jsonl.gz", sp_lookup_path=None):
      """Compute reweighted kernel matrix from streamed JSONL input."""

      K_weighteds = np.zeros((len(midpoints_ids), len(knots_ids)))

      # For fast lookup
      midpoint_set = set(midpoints_ids)
      knot_set = set(knots_ids)
      if sp_lookup_path == None:
        # Build lookup dictionary of relevant shortest paths
        sp_lookup = {}

        print("📦 Indexing valid shortest paths...")
        with gzip.open(path_to_jsonl, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Indexing JSONL lines"):
                sp_info = json.loads(line)
                i, j = sp_info["origin_id"], sp_info["destination_id"]

                # Only keep if (midpoint ↔ knot) pair
                if (i in midpoint_set and j in knot_set) or (j in midpoint_set and i in knot_set):
                    key = tuple(sorted((i, j)))
                    sp_lookup[key] = sp_info

        print(f"✅ Indexed {len(sp_lookup)} valid pairs.")

      else:
        sp_lookup = pickle.load(open(sp_lookup_path, "rb"))
        print(f"✅ Loaded {len(sp_lookup)} valid pairs from {sp_lookup_path}.")

      # Compute kernel matrix
      print("🧮 Computing reweighted kernel matrix...")
      for i_idx, midpoint_id in enumerate(tqdm(midpoints_ids, desc="Rows")):
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
              if np.isinf(sp):
                K = 0  # no connection
              else:
                reweight = np.prod(weights)
                if kernel_function == 'Triangular':
                    K = (reweight ** 0.5) * max(1 - abs(sp / h), 0)
                elif kernel_function == 'Epanechnikov':
                    K = (reweight ** 0.5) * max((3/4)*(1-(sp/h)**2),0)
                elif kernel_function == 'Quartic':
                    if sp/h <1:
                      K = (reweight ** 0.5) * (15/16)*((1-(sp/h)**2)**2)
                    else:
                      K= 0
              K_weighteds[i_idx, j_idx] = K

      return K_weighteds

  for kernel_function in kernel_functions:
    try:
      filename = output_folder+f'/2_K_weighted_k{k}_h{h}_{kernel_function}.pkl'
      with open(filename, 'rb') as f:
        K_weighted = pickle.load(f)
      print("✅ Kernel matrix shape:", K_weighted.shape)
    except:
      # After defining midpoints_ids and knots_ids:
      print('Here, using kernel:', kernel_function)
      K_weighted = compute_reweighted_kernels(midpoints_ids, knots_ids, kernel_function=kernel_function, sp_lookup_path='0_sp_lookups.pkl')
      print("✅ Kernel matrix shape:", K_weighted.shape)

      filename = output_folder+f'/2_K_weighted_k{k}_h{h}_{kernel_function}.pkl'
      with open(filename, 'wb') as f:
        pickle.dump(K_weighted, f)
      print("✅ Kernel matrix shape:", K_weighted.shape)

    # ==============================================
    # Step 3: Fit NPC Model with Reweighted Kernel
    # ==============================================
    # === Prepare Covariate Matrix ===
    # Standardize log(traffic volume)
    traffic_volume_log = np.log(data['traffic_volume'].values)
    traffic_volume_log = (traffic_volume_log - traffic_volume_log.mean()) / traffic_volume_log.std()
    data['log_traffic_volume_std'] = traffic_volume_log

    # Build X matrix with continuous and dummy covariates
    X = data[['log_traffic_volume_std']].values

    with pm.Model() as model:
      # Priors
      beta0 = pm.Normal("beta0", mu=0, sigma=1)  # Tightened priors
      var_psi = pm.Gamma("var_psi", alpha=2, beta=2)  # variance
      sigma_psi = pm.Deterministic("sigma_psi", pm.math.sqrt(var_psi))
      psi = pm.Normal("psi", mu=0, sigma=sigma_psi, shape=len(knots_ids))

      # Spatial random effects with reweighted kernel
      Z = pm.math.dot(K_weighted, psi)

      # Expected crashes (log-link with offset)
      log_lambda = pm.math.log(data['length_m'].values) + pm.math.log(data['years'].values) + beta0 + Z
      lambda_ = pm.Deterministic("lambda_", pm.math.exp(log_lambda))

      # Likelihood
      y_obs = pm.Poisson("y_obs", mu=lambda_, observed=data['observed_crashes'].values)

      # Sample
      trace = pm.sample(draws=3000,
                        tune=1000,
                        chains=2,
                        target_accept=0.95,         # More cautious steps
                        max_treedepth=15,
                        random_state=42,
                        return_inferencedata=True,
                        idata_kwargs={"log_likelihood": True})


    # Save trace
    az.to_netcdf(trace, output_folder+f"/2_trace_k{k}_h{h}_{collision_type}_{kernel_function}.nc")

    idata = az.from_netcdf(output_folder+f"/2_trace_k{k}_h{h}_{collision_type}_{kernel_function}.nc")
    # WAIC
    waic = az.waic(idata)
    print("WAIC:", waic)

    # Save to file
    with open(output_folder+f"/2_waic_k{k}_h{h}_{collision_type}_{kernel_function}.pkl", "wb") as f:
        pickle.dump(waic, f)

    # LOO
    loo = az.loo(idata)
    print("LOO:", loo)

    # Save to file
    with open(output_folder+f"/2_loo_k{k}_h{h}_{collision_type}_{kernel_function}.pkl", "wb") as f:
        pickle.dump(loo, f)


    # Summarize the posterior
    # Posterior Summary
    summary = az.summary(trace, var_names=["beta0", "var_psi"], round_to=4, hdi_prob=0.95)

    # Output stats
    summary_stats = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].copy()
    summary_stats.to_csv(output_folder+f'/5_summary_stats_k{k}_h{h}_{collision_type}_{kernel_function}.csv')
    print(summary_stats)