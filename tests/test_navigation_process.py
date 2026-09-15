"""All four applications, isolated operational consumers and real transports."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from dcmn.archive import ArchiveReader
from dtest.isolation import isolated_env, violations

ROOT=Path(__file__).resolve().parents[1]
PROFILE=ROOT/'assets/execution_profiles/dry-run.json'


@pytest.mark.parametrize('real_simulator',[False,True])
def test_dynamic_route_process_chain(tmp_path,real_simulator):
    processes=[]
    logs=[]
    def start(args,env=None):
        log=tmp_path/f'process-{len(processes)}.log'
        with log.open('w') as out:
            process=subprocess.Popen([sys.executable,*map(str,args)],cwd=ROOT,env=env,
                                     stdout=out,stderr=subprocess.STDOUT)
        processes.append(process); logs.append(log)
        return process
    try:
        with ExitStack() as stack:
            if real_simulator:
                from dtest.process_harness import DsimProcessHarness
                harness=stack.enter_context(DsimProcessHarness(tmp_path,map_path=ROOT/'assets/maps/maze_012.txt'))
                instance,report=harness.id,harness.report_dir
                algorithm='ground-plane-baseline'
                goal='5.5,1.5'
            else:
                instance='nav-chain-'+uuid.uuid4().hex[:8]
                report=tmp_path/'run'
                algorithm='lidar-baseline'
                goal='8,2'
            env=isolated_env(tmp_path/'isolation')
            dry=start([ROOT/'apps/dway/dway.py','--id',instance,'--mode','dynamic','--dry-run',
                       '--no-ui','--execution-profile',PROFILE,'--timeout','10'],env)
            alg=start([ROOT/'apps/dalg/dalg.py','--id',instance,'--profile',algorithm,'--no-ui',
                       '--timeout','12'],env)
            nav=start([ROOT/'apps/dnav/dnav.py','--id',instance,'--goal',goal,'--no-ui',
                       '--execution-profile',PROFILE,'--timeout','12'],env)
            if not real_simulator:
                time.sleep(.5)  # exercise late provider startup
                start([ROOT/'dtest/provider.py','--id',instance,'--report-dir',report,
                       '--sensors','scan','--path','2,0;2,2.5;2,2.5','--timeout','25'])
            for process in (dry,alg,nav):
                assert process.wait(timeout=35)==0, '\n'.join(p.read_text() for p in logs)
            assert violations(tmp_path/'isolation')==[]
            summary=json.loads((report/'dnav/summary.json').read_text())
            assert summary['attempts']>0
            assert summary['navigation']['sequence']>0
            reader=ArchiveReader(report/'dway/archive')
            assert reader.validate()['complete']
            observed=[e for e in reader.events() if e['type']=='navigation.observed']
            assert any(e['data']['navigation'].get('sequence',0)>0 for e in observed)
            assert all(e['data']['execution']['commanded_target'] is None for e in observed)
            assert all(e['data']['execution']['owns_control'] is False for e in observed)
            assert (report/'dway/archive/report.html').exists()
            if real_simulator:
                # The dry-run never touched the vehicle: the harness still owns its lease.
                owner = harness.read_status().get('control.owner', '')
                assert 'dway' not in str(owner), owner
            else:
                # Clear observed space with a live evidence producer must reach admission.
                states = {e['data']['execution']['state'] for e in observed}
                assert 'READY' in states, states
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
