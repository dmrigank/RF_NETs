# RF_NETs
Multi-fidelity operator-net Richardson-Net approach


Data gen:

python phase1_generate_heat_periodic_AB.py \
  --out_dir heat_phase1_data \
  --Ns 50 100 200 \
  --num_samples 2000 \
  --K 16 \
  --t_snap 0.05 0.10 0.20 \
  --alpha_min 0.5 --alpha_max 2.0 \
  --coeff_scale 1.0 \
  --r_target 0.4 \
  --dtype_store float32 \
  --sanity_check


Baselines:

python phase2_baselines_heat_periodic_AB.py \
  --in_dir heat_phase1_data \
  --out_dir heat_phase2_baselines \
  --metric L2_mean \
  --make_snapshots \
  --snapshot_sample_idx 0


RFNET implementation + training:

python phase3_train_richardsonnet_heat_periodic_AB.py \
  --npz heat_phase1_data/heat_periodic_AB_CN2_FD2_N100_K16_S2000_T0.2.npz \
  --out_dir heat_phase3_runs/N100_rich_sup \
  --method rich_sup \
  --epochs 2000 \
  --batch_size 64 \
  --q_factor 4 \
  --lambda_rich 1.0 \
  --num_rich 32 \
  --make_snapshots
