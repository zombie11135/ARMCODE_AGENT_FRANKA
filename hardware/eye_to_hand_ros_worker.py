"""Read-only ROS bridge, sent to the existing Docker container as Python source.

Subscribes joint states; the ONLY service used is /compute_fk. No controller,
action client, FCI connection, gripper, or motion API is created.
"""
import json
import math
import os
import select
import sys
import time


def main():
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from moveit_msgs.msg import RobotState
    from moveit_msgs.srv import GetPositionFK

    names = [f"fr3_joint{i}" for i in range(1, 8)]
    rclpy.init()
    node = rclpy.create_node(f"eye_to_hand_read_only_{os.getpid()}")
    client = node.create_client(GetPositionFK, "/compute_fk")
    last_sent = 0.
    pending = {}

    def emit(data):
        print("EYE_TO_HAND=" + json.dumps(data, allow_nan=False), flush=True)

    def joints(msg):
        nonlocal last_sent
        now = time.monotonic()
        if now - last_sent < .025:
            return
        q = dict(zip(msg.name, msg.position))
        dq = dict(zip(msg.name, msg.velocity))
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = node.get_clock().now().nanoseconds * 1e-9 - stamp
        if not -.1 <= age <= .5:
            return
        if not all(j in q and j in dq and math.isfinite(q[j]) and math.isfinite(dq[j]) for j in names):
            return
        last_sent = now
        emit({"kind": "joints", "received_at": now, "ros_stamp_s": stamp,
              "ros_age_s": age, "q": [q[j] for j in names], "dq": [dq[j] for j in names]})

    subscription = node.create_subscription(JointState, "/joint_states", joints,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

    def fk_done(future, identifier):
        pending.pop(identifier, None)
        try:
            response = future.result()
            if response.error_code.val != 1 or len(response.pose_stamped) != 1:
                raise RuntimeError(f"FK failed: {response.error_code.val}")
            stamped = response.pose_stamped[0]
            if stamped.header.frame_id.lstrip("/") != "fr3_link0":
                raise RuntimeError("Unexpected FK reference frame")
            p = stamped.pose
            emit({"kind": "fk", "id": identifier,
                  "xyz": [p.position.x, p.position.y, p.position.z],
                  "xyzw": [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]})
        except Exception as exc:
            emit({"kind": "fk", "id": identifier, "error": str(exc)})

    buffer = b""
    running = True
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            if select.select([sys.stdin], [], [], 0)[0]:
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    break  # Owner has exited; stop this subscriber only.
                buffer += chunk
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    request = json.loads(raw)
                    if request.get("op") == "quit":
                        running = False
                        break
                    identifier = request.get("id")
                    q = request.get("q", [])
                    if request.get("op") != "fk" or len(q) != 7 or not all(math.isfinite(x) for x in q):
                        raise ValueError("Only finite seven-joint FK requests are accepted")
                    if not client.service_is_ready():
                        emit({"kind": "fk", "id": identifier, "error": "/compute_fk unavailable; start real.sh"})
                        continue
                    req = GetPositionFK.Request()
                    req.header.frame_id = "fr3_link0"
                    req.fk_link_names = ["fr3_link8"]
                    req.robot_state = RobotState(joint_state=JointState(name=names, position=q), is_diff=True)
                    future = client.call_async(req)
                    pending[identifier] = future
                    future.add_done_callback(lambda f, i=identifier: fk_done(f, i))
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
