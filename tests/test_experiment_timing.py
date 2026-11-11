"""No devices: known-duration clocks and existing Session API doubles."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from hardware.fr3_worker import install_timing
from runtime.experiment_timing import phases, export
from runtime.timing import task_timing, timing_stage, record_worker_timing


def test_worker_separates_planner_and_controller_and_preserves_original_calls(monkeypatch):
    clock=[10.]
    monkeypatch.setattr('hardware.fr3_worker.time.monotonic',lambda:clock[0])
    response=NS(motion_plan_response=NS(planning_time=1.9,error_code=NS(val=1)))
    outcome=NS(status=4,result=NS(error_code=0))
    def call(srv,name,request,timeout):
        clock[0]+=2.3
        return response
    def wait(future,timeout,monitor):
        clock[0]+=11.5 if monitor else .2
        return outcome
    session=NS(call=Mock(side_effect=call), wait=Mock(side_effect=wait),prepare=Mock(return_value=('i','t','traj')))
    original_call,original_wait=session.call,session.wait
    result={}
    install_timing(session,result)
    request, future=object(),object()
    assert session.call('srv','/plan_kinematic_path',request) is response
    assert session.wait(future,30,True) is outcome
    assert session.wait(future,3,False) is outcome
    original_call.assert_called_once_with('srv','/plan_kinematic_path',request,15)
    assert original_wait.call_count==2
    assert [s['phase'] for s in result['timing']]==['MOTION_PLAN','MOVE']
    assert [s['duration_s'] for s in result['timing']]==pytest.approx([2.3,11.5])
    assert result['timing'][0]['attributes']['server_planning_s']==1.9
    assert session.prepare()==('i','t','traj')


@pytest.mark.parametrize('error',[RuntimeError('controller failure'),KeyboardInterrupt()])
def test_worker_failure_still_records_and_reraises_exact_error(error):
    session=NS(call=Mock(), wait=Mock(side_effect=error),prepare=Mock())
    result={}
    install_timing(session,result)
    with pytest.raises(type(error)) as raised:
        session.wait(object(),10,True)
    assert raised.value is error
    assert result['timing'][0]['status']=='error'


def sample(start,duration,phase='MOVE',status='ok'):
    return dict(name='worker',phase=phase,start_monotonic_s=start,duration_s=duration,status=status,attributes={})


def test_import_clock_alignment_and_handoff_and_missing_stage(tmp_path):
    path=tmp_path/'task.json'
    with task_timing('two objects',path) as recorder:
        origin=recorder.monotonic_started
        for request,start in [('a',1),('b',7)]:
            with timing_stage('机械臂调用',detail='move'):
                record_worker_timing(dict(request_id=request,timing_clock='clock',operation='move',
                    timing=[sample(origin+start,2)]),'clock')
    report=json.loads(path.read_text())
    assert report['phases']['MOVE']['wall_s']==4
    assert report['phases']['HANDOFF']['wall_s']==4
    assert report['phases']['VERIFY']['wall_s'] is None
    assert report['phases']['VERIFY']['coverage']=='not_observed'
    assert list((tmp_path/'timing').glob('*.csv'))


def test_mismatched_clock_retains_duration_without_fabricated_handoff(tmp_path):
    path=tmp_path/'task.json'
    with task_timing('clock',path):
        for i in range(2):
            record_worker_timing(dict(request_id=str(i),timing_clock='other',timing=[sample(i*10,2)]),'host')
    data=json.loads(path.read_text())
    assert data['phases']['MOVE']['work_s']==4
    assert data['phases']['MOVE']['wall_s'] is None
    assert data['phases']['HANDOFF']['wall_s'] is None


def test_missing_worker_packet_breaks_handoff_chain(tmp_path):
    path=tmp_path/'task.json'
    with task_timing('missing',path) as rec:
        for i in range(3):
            with timing_stage('机械臂调用',detail='move'):
                if i!=1:
                    record_worker_timing(dict(request_id=str(i),timing_clock='c',
                        timing=[sample(rec.monotonic_started+5*i,2)]),'c')
    assert json.loads(path.read_text())['phases']['HANDOFF']['wall_s'] is None


def test_overlapping_perception_spans_are_unioned_not_summed():
    report=dict(duration_s=12,status='ok',spans=[
        dict(id=1,name='目标身份识别',detail='',start_s=0,duration_s=8,status='ok'),
        dict(id=2,name='SAM3分割与遮挡关系构建',detail='',start_s=3,duration_s=8,status='ok')])
    p=phases(report)['PERCEIVE']
    assert p['wall_s']==11 and p['work_s']==16


def test_export_stratifies_failures_recovery_and_dry_runs_and_deduplicates(tmp_path):
    paths=[]
    for i,(status,recovery,real,duration) in enumerate([
            ('ok',False,True,10),('ok',False,True,20),('ok',True,True,30),
            ('failed',False,True,8),('ok',False,False,1)]):
        report=dict(run_id=str(i),goal='a then b',status=status,duration_s=duration,spans=[],
            metadata=dict(experiment='experiment',execute_robot=real,hardware_profile={'arm':{'stack':'real'}}),
            recovery_events=[{'kind':'retry'}] if recovery else [])
        report['phases']=phases(report)
        path=tmp_path/f'{i}.json'; path.write_text(json.dumps(report));paths.append(path)
    result=export(paths+[paths[0]],tmp_path/'out','experiment')
    assert result['trial_count']==4
    rows=[s for s in result['summary'] if s['phase']=='TOTAL']
    ok=next(s for s in rows if s['outcome']=='ok_without_recovery')
    assert (ok['mean_s'],ok['p50_s'],ok['p90_s'])==(15,15,19)
    assert {r['outcome'] for r in rows}=={'ok_without_recovery','ok_with_recovery','failed_without_recovery'}
    assert all(s['mean_s'] is None for s in result['summary'] if s['phase']=='VERIFY')


def test_false_controller_result_is_not_a_successful_motion_boundary():
    session=NS(call=Mock(),wait=Mock(return_value=NS(status=6,result=NS(error_code=-1))),prepare=Mock())
    result={}; install_timing(session,result)
    assert session.wait(None,10,True).status==6
    assert result['timing'][0]['status']=='failed'


def test_early_real_failure_is_exported_but_fake_stack_is_not(tmp_path):
    paths=[]
    for index,profile in enumerate([None, {'arm': {'stack':'real'}}, {'arm': {'stack':'fake'}}]):
        data=dict(run_id=str(index),goal='task',status='error',duration_s=3,spans=[],
            metadata=dict(experiment='e',execute_robot=True,configured_stack='real',hardware_sha256='same'),
            recovery_events=[])
        if profile is not None:
            data['metadata']['hardware_profile']=profile
        data['phases']=phases(data)
        path=tmp_path/f'{index}.json';path.write_text(json.dumps(data));paths.append(path)
    result=export(paths,tmp_path/'out','e')
    assert result['trial_count']==2 and len(result['groups'])==1


def test_launcher_keeps_motion_opt_in_and_isolates_trial_artifacts(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    scripts=tmp_path/'scripts';scripts.mkdir()
    source=Path(__file__).resolve().parents[1]/'scripts/run_fr3_timing_trial.sh'
    shutil.copy(source,scripts/source.name)
    (tmp_path/'run_fr3_agent.sh').write_text('#!/bin/bash\nprintf \'%s\\n\' "$@"\n')
    env=dict(os.environ,ARMCODE_TIMING_EXPERIMENT='stub-only')
    command=['bash',str(scripts/source.name),'trial01']
    first=subprocess.run(command,check=True,capture_output=True,text=True,env=env).stdout
    second=subprocess.run(command+['--execute_robot'],check=True,capture_output=True,text=True,env=env).stdout
    assert '--execute_robot' not in first.splitlines()
    assert '--execute_robot' in second.splitlines()
    assert len(list((tmp_path/'runs').iterdir()))==2
    assert '西瓜霜' in first and '抗病毒口服液' in first


def test_hardware_config_and_conditions_are_captured_without_credentials(tmp_path, monkeypatch):
    from runtime.timing import experiment_metadata
    config=tmp_path/'hardware.yaml'
    config.write_text('arm: {stack: real}\nperception: {grasp_backend: graspnet_baseline}\n')
    monkeypatch.setenv('ARMCODE_TIMING_TRIAL','trial01')
    result=experiment_metadata(NS(hardware_config_path=str(config),api_key='do-not-export',arm_speed_scaling=.2))
    assert result['configured_stack']=='real' and result['configured_grasp_backend']=='graspnet_baseline'
    assert result['trial']=='trial01' and result['conditions']['arm_speed_scaling']==.2
    assert 'do-not-export' not in json.dumps(result)
