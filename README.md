# Jetpose

**Implementation of the FoundationPose library for Jetson Orin NX with live inference using Realsense D435i.**
### On local conda environment.

## Install Dependencies

### Step-by-Step Instructions
Follow the commented instructions in `build_all.sh` step by step. **Do not run the whole script at once.** Instead, copy and paste each command manually for ease of debugging.

### Librealsense Dependencies
1. Run `librealsensesSDK_install.sh` to install the necessary dependencies for librealsense.
2. To enable `pyrealsense2` support, copy the required `.so` files to the folder where the script will run in the Jetpose directory. Example:

```bash
cp ~/librealsense_build/librealsense-master/build/release/pyrealsense2.cpython-310-aarch64-linux-gnu.so.2.55.1 ~/Jetpose/FoundationPose/pyrealsense2.so
cp ~/librealsense_build/librealsense-master/build/release/librealsense2.so.2.55.1 ~/Jetpose/FoundationPose/librealsense2.so
cp ~/librealsense_build/librealsense-master/build/release/librealsense2-gl.so.2.55.1 ~/Jetpose/FoundationPose/librealsense2-gl.so
```

---

## Run

1. Open `run_live.py` and set the path for the `.obj` file.
2. Run the script using:

```bash
python run_live.py
```

---

## Notes

- During development, I couldn't use the cuDNN backend in PyTorch. As a workaround, I disabled it using:

```python
# torch.backends.cudnn.enabled = False
```

This results in slow inference, approximately **1 inference per second**.

- Help is needed to enable cuDNN backend for PyTorch on Jetson Orin NX.

## If you find this repository helpful, please consider giving it a star.
---

