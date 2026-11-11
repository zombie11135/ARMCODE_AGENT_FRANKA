"""Headless EGL selection before importing robosuite; no driver changes."""
import os
# Load Python's matching expat before Mesa can load a different system version.
import xml.parsers.expat  # noqa: F401


def configure_renderer():
    os.environ.setdefault('MUJOCO_GL', 'egl')
    if os.environ['MUJOCO_GL'] != 'egl' or 'MUJOCO_EGL_DEVICE_ID' in os.environ:
        return
    from mujoco.egl import egl_ext as egl
    from OpenGL.error import GLError

    for index, device in enumerate(egl.eglQueryDevicesEXT()):
        display = egl.eglGetPlatformDisplayEXT(egl.EGL_PLATFORM_DEVICE_EXT, device, None)
        try:
            initialized = egl.eglInitialize(display, None, None)
        except GLError:
            continue
        if initialized:
            egl.eglTerminate(display)
            os.environ['MUJOCO_EGL_DEVICE_ID'] = str(index)
            return
    raise RuntimeError('No working EGL device. Configure MUJOCO_GL=osmesa or a working EGL device.')
