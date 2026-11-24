"""colcon test hook: the original dobot_description must expand, assemble and kinematically
check out in every launch configuration. WARN-level findings are known original-model
defects (documented in docs/PHASE1_VERIFICATION.md) and do not fail the build."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))


def _pkg():
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory('dobot_description')
    except Exception:  # noqa: BLE001  (source tree fallback)
        return os.path.abspath(os.path.join(HERE, '..', '..', 'dobot_description'))


def test_model_verification():
    import verify_model as v
    pkg = _pkg()
    v.results.clear()
    v.check_configs(pkg)
    v.check_clean_equivalence(pkg)
    v.check_real_xacro(pkg)
    m = v.load_model(pkg, DOF=4, tool='gripper', use_camera='false')
    v.check_meshes(pkg, m)
    v.check_limits(pkg, m)
    v.check_kinematics(m)
    v.check_workspace(m)
    v.check_alignment(m)
    fails = [r for r in v.results if r[0] == 'FAIL']
    assert not fails, fails
