# use u and v to code the emulator and make a loss function based on LE in the next time step?
# probably more natural to use xarray instead of h5py in order (at least) to read the data...
# should be nice to extract inner + outter step functions into one so that it can be used here
# and then there is maybe going to be some forcing
# in the online step one would call the entire run_simulation function...

import os

import h5py
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array
from utils import HybridObukhovModel, NeuralNetwork

import abcconfigs.class_model as cm
import abcmodel


def load_data(key: Array, ratio: float = 0.8) -> tuple[Array, ...]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../data/dataset.h5")

    # 1. READ ALL VARIABLES (ENTIRE STATE)
    features = []
    target = None
    target_key_suffix = "le"  # We look for the variable ending in 'le' (Latent Energy)

    def collector(name, node):
        # We only care about Datasets (arrays), not Groups (folders)
        if isinstance(node, h5py.Dataset):
            if name == "time":
                return

            # Read data: Shape (Num_Ensembles, Time)
            data = jnp.array(node)

            # Ensure it is (N, 1, T) so we can stack features on axis 1
            if data.ndim == 2:
                data = jnp.expand_dims(data, axis=1)

            features.append(data)

            # Identify target variable if it matches our suffix (e.g., 'land/le')
            nonlocal target
            if name.endswith(target_key_suffix):
                target = data

    with h5py.File(file_path, "r") as f:
        # Recursively visit every node in the file
        f.visititems(collector)

    if target is None:
        raise ValueError(
            f"Target variable ending in '{target_key_suffix}' not found in H5 file."
        )

    # Stack all features to form X: (N, Num_Features, T)
    # Target Y is just the specific variable we want to predict
    x_full = jnp.concatenate(features, axis=1)
    y_full = target

    # 2. TIME SHIFTING
    # We use state at `t` (x) to predict target at `t+1` (y)
    x = x_full[..., :-1]  # Drop last timestep
    y = y_full[..., 1:]  # Drop first timestep

    # 3. SPLIT TRAIN/TEST
    num_ensembles = x.shape[0]
    perm_idxs = jax.random.permutation(key, num_ensembles)
    split_idx = int(ratio * num_ensembles)

    train_idxs = perm_idxs[:split_idx]
    test_idxs = perm_idxs[split_idx:]

    x_train, x_test = x[train_idxs], x[test_idxs]
    y_train, y_test = y[train_idxs], y[test_idxs]

    # 4. NORMALIZATION (Standard Score)
    # Crucial: Calculate stats ONLY on training data to avoid information leakage

    # Normalize X (Entire State)
    # We average over samples (axis 0) and time (axis 2) to get one mean/std per feature
    x_mean = jnp.mean(x_train, axis=(0, 2), keepdims=True)
    x_std = jnp.std(x_train, axis=(0, 2), keepdims=True)
    # Prevent division by zero for constant variables
    x_std = jnp.where(x_std < 1e-6, 1.0, x_std)

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std

    # Normalize Y (Target)
    y_mean = jnp.mean(y_train)
    y_std = jnp.std(y_train)

    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Return data + the stats needed to un-normalize predictions later
    return x_train, x_test, y_train, y_test, y_mean, y_std


def load_model(key: Array):
    rad_model = abcmodel.rad.StandardRadiationModel(
        **cm.standard_radiation.model_kwargs
    )
    land_model = abcmodel.land.JarvisStewartModel(**cm.jarvis_stewart.model_kwargs)

    # definition od the hybrid model
    mkey, hkey = jax.random.split(key)
    psim_net = NeuralNetwork(rngs=nnx.Rngs(mkey))
    psih_net = NeuralNetwork(rngs=nnx.Rngs(hkey))
    hybrid_surface = HybridObukhovModel(psim_net, psih_net)

    mixed_layer_model = abcmodel.atmos.mixed_layer.BulkMixedLayerModel(
        **cm.bulk_mixed_layer.model_kwargs
    )
    cloud_model = abcmodel.atmos.clouds.CumulusModel()
    atmos_model = abcmodel.atmos.DayOnlyAtmosphereModel(
        surface_layer=hybrid_surface,
        mixed_layer=mixed_layer_model,
        clouds=cloud_model,
    )

    return abcmodel.ABCoupler(rad=rad_model, land=land_model, atmos=atmos_model)


def train(model, x: Array, y: Array, y_mean: Array, y_std: Array):
    # time settings: should be the same as the
    # ones that were used to generate the dataset
    inner_dt = 60.0
    outter_dt = 60.0 * 30
    tstart = 6.5
    inner_tsteps = int(outter_dt / inner_dt)

    print("training...")
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    def loss_fn(model, x, y):
        pred = abcmodel.integration.outter_step(
            x,
            x.t,
            coupler=model,
            inner_dt=inner_dt,
            inner_tsteps=inner_tsteps,
            tstart=tstart,
        )
        pred_le = (pred.land.le - y_mean) / y_std  # type: ignore
        return jnp.mean((pred_le - y) ** 2)

    @jax.jit
    def update(model, optimizer, x, y):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
        optimizer.update(grads)
        return loss

    for step in range(2000):
        loss = update(model, optimizer, x, y)
        if step % 500 == 0:
            print(f"  step {step}, loss: {loss:.6f}")

    return model


def main():
    key = jax.random.PRNGKey(42)
    data_key, model_key = jax.random.split(key)
    x_train, x_test, y_train, y_test, y_mean, y_std = load_data(data_key)
    model = load_model(model_key)
    model = train(model, x_train, y_train, y_mean, y_std)


if __name__ == "__main__":
    main()
