# AnyPlace real-data stage 1 (offline only)

> 源码发布说明：以下为已有模块和实验环境的使用记录。绝对路径需要按本机修改，模型、外部源码、照片、标定与实测 manifest 不随仓库分发。新成员先阅读 [SETUP](../docs/SETUP.md) 与 [第三方依赖](../THIRD_PARTY.md)。

This entry prepares recorded aligned RGB-D, invokes the unmodified upstream
AnyPlace inference function, and exports point clouds plus a placement proposal.
It contains no API that enables motion, operates a gripper, clears errors, or
changes collision thresholds.

## Stage 2: tool calibration, multiview geometry and candidate filtering

The geometry, fitting, inference and filtering tools are offline.
`record_tcp_teach_poses` connects to xArm only to call read APIs after the
operator manually positions the arm. `xarm_teach_mode` is a separate, explicit
operator helper that can enable the motors and change controller mode/state; it
requires confirmation and never sends a pose, clears an error, writes a TCP
offset or operates the GT200.

### Enter or leave xArm joint teaching mode

Keep the emergency stop reachable, remove obstacles and support any unstable
payload before entering teaching mode. The helper refuses to proceed when the
arm is moving or the controller reports an error/warning.

```bash
/home/wpy/.conda/envs/armcode_agent/bin/python -m anyplace_real.xarm_teach_mode \
  enter --xarm-ip 192.168.1.237
```

It asks the physically present operator to type `TEACH`, then performs only
`motion_enable(True)`, `set_mode(2, detection_param=0)` and `set_state(0)`.
Check the current status without changing it:

```bash
/home/wpy/.conda/envs/armcode_agent/bin/python -m anyplace_real.xarm_teach_mode \
  status --xarm-ip 192.168.1.237
```

After teaching, stop the arm and return the controller to position mode without
sending a position command:

```bash
/home/wpy/.conda/envs/armcode_agent/bin/python -m anyplace_real.xarm_teach_mode \
  leave --xarm-ip 192.168.1.237
```

### Low-cost GT200 pivot teaching

No precision fixture is required for a first audit. Use a rigid straight pen,
wooden skewer or narrow metal rod, paper, tape and a ruler:

1. Grip/tape the rod rigidly on the GT200 centreline. Mark the distance from
   the desired grasp-centre TCP to the rod tip; the fitted result is the tip,
   so this measured overhang must later be removed along the defined tool axis.
2. Tape paper to a rigid table and make one small cross or shallow dimple. A
   dimple is safer than trying to maintain contact on a flat mark.
3. In xArm Studio/manual mode, bring the tip to exactly that same point with
   negligible force. Keep the tip position fixed while changing wrist
   orientation by at least 30 degrees overall. Avoid near-identical poses.
4. Stop at 8--12 orientations and press Enter in the read-only recorder. Keep
   one or two additional poses out of the fit for validation.
5. Run the audit and inspect RMS/max residual. This finds the tip position but
   not the TCP orientation; define orientation from the GT200 mechanical axes.

Use a soft pen/skewer rather than a sharp metal point, work at low manual speed,
keep the area clear and keep the emergency stop reachable. Paper/tape teaching
is suitable for an initial 2--4 mm audit, not tight insertion. Repeat with a
rigid commercial pointer or surveyed fixture before sub-millimetre work.

Read-only pose recording (the controller TCP offset must still be zero):

```bash
cd /home/wpy/pythonproject/armcode-agent
/home/wpy/.conda/envs/armcode_agent/bin/python -m anyplace_real.record_tcp_teach_poses \
  --xarm-ip 192.168.1.237 --count 10 \
  --output runs/anyplace_real_stage2/tcp/teaching_poses.json
```

Offline fitting and residual report:

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.tcp_calibration_audit \
  --manifest runs/anyplace_real_stage2/tcp/teaching_poses.json \
  --output-dir runs/anyplace_real_stage2/tcp/audit
```

The recorder leaves `tcp_orientation_rpy_rad=null`, so the audit deliberately
reports position-only and does not create an authoritative full transform.
After defining and independently checking the GT200 tool axes, put that RPY in
the manifest and rerun. Review the output before any separate controller write;
none of these tools calls `set_tcp_offset` or `save_conf`.

### Saved RGB-D multiview fusion

Copy `config/multiview_manifest.example.json`, replace all image, mask and
flange-pose values, set `example_only=false`, and supply measured medicine-box
dimensions and the fitted table height. Use `visible_surface_mask.npy`, not the
eroded AnyGrasp mask.

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.multiview_fusion \
  --manifest anyplace_real/config/my_multiview_manifest.json \
  --output-dir runs/anyplace_real_stage2/multiview
```

The result keeps `observed_fused_cloud` separate from
`complete_collision_proxy`: unobserved cuboid faces are explicitly marked as a
measured-dimensions proxy, never as depth measurements. The current saved
snapshot collection is not suitable input because its captures are all from
essentially the same wrist pose.

### Per-grasp held-object relation

For each successful grasp, estimate the same labelled box frame in at least
three post-grasp images and pair it with the synchronized TCP pose. Copy
`config/held_object_manifest.example.json`, replace the matrices, and run:

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.held_object_relation \
  --manifest anyplace_real/config/my_held_object_manifest.json \
  --output-dir runs/anyplace_real_stage2/held
```

It computes `T_tcp_object = inverse(T_base_tcp) @ T_base_object`, averages the
relations on SE(3), and rejects excessive translation or rotation scatter. A
wrist camera can estimate the held pose if the package remains visible, but
moving the arm does not create new relative views because camera and gripper
move rigidly together. Use an external camera if more held-object faces are
needed. The result is per grasp and must be invalidated after slip or regrasp.

### Filter all AnyPlace candidates

```bash
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.candidate_filter \
  --predictions runs/anyplace_real_stage1/new_capture/anyplace_stage1_relative_pose_prediction_real.npy \
  --target-proxy runs/anyplace_real_stage2/multiview/complete_collision_proxy.npy \
  --support runs/anyplace_real_stage1/new_capture/base_obj_pcd.npy \
  --T-base-object-input runs/anyplace_real_stage2/multiview/T_base_object_input.npy \
  --T-tcp-object runs/anyplace_real_stage2/held/T_tcp_object.npy \
  --output-dir runs/anyplace_real_stage2/candidate_filter
```

This evaluates all predictions, projects each complete proxy to the measured
support plane, checks support containment/contact/centre and optional obstacle
clearance, then composes pre-place/place/retreat TCP matrices. The output always
keeps `execution_allowed=false`: xArm/GT200 IK and swept collision validation
remain a separate required stage.

Prepare the saved real snapshot with the existing robot/perception environment:

```bash
/home/wpy/.conda/envs/armcode_agent/bin/python -m anyplace_real.prepare_real \
  --manifest anyplace_real/config/real_saved_xarm_snapshot.json \
  --output-dir runs/anyplace_real_stage1/real_saved_xarm_snapshot
```

Run AnyPlace in its existing environment. The extra paths expose the CUDA
headers already installed in that same environment; no package downgrade is
performed:

```bash
PATH=/home/wpy/.conda/envs/anyplace/bin:$PATH \
CPATH=/home/wpy/.conda/envs/anyplace/targets/x86_64-linux/include \
LIBRARY_PATH=/home/wpy/.conda/envs/anyplace/targets/x86_64-linux/lib \
LD_LIBRARY_PATH=/home/wpy/.conda/envs/anyplace/targets/x86_64-linux/lib \
/home/wpy/.conda/envs/anyplace/bin/python -m anyplace_real.run_inference \
  --input-dir runs/anyplace_real_stage1/real_saved_xarm_snapshot \
  --n-refine-iters 10 --init-k 5 --seed 0
```

For the saved real snapshot, both clouds are back-projected in the camera frame
and then transformed into the capture-time legacy xArm base frame so AnyPlace
receives gravity-up coordinates. The returned transform is still only an object-
motion proposal: it is not a robot TCP command and must not be executed as one.
