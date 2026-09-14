"""dnav's report page: dark, built from the summary alone, never raising into the run."""
import json

from dnav.report import write_html, write_report
from dcmn import theme


def test_report_page_is_dark_and_says_what_was_planned(tmp_path) -> None:
    summary = dict(instance='area1', session_id='s1', planner='astar', attempts=3, plans=2, health='ok',
                   goal=dict(position=[5.0, 2.0]),
                   route=dict(status='ok', reason='', length_m=4.2, cost=5.1),
                   detour=dict(length_ratio=1.1, cost_ratio=1.2),
                   navigation=dict(stop_generation=1, geometry_revision=2, clearance=dict(
                       eligible=True, permission='plan', reason='planned route trusted', distance_m=4.0,
                       evidence_check=dict(eligible=False, reason='unknown cells'))),
                   status_counts=dict(ok=2, stale_map=1),
                   statuses=[dict(sim_time_s=1.0, status='stale_map', planner='astar', map_revision=0, reason='<wait>')],
                   map=dict(sources={'scan': dict(age_s=0.5, rate_hz=1.0)}),
                   recording=dict(state='finalized', complete=True), policy=dict(name='default'))
    write_report(tmp_path, summary=summary)
    page = (tmp_path / 'report.html').read_text()
    assert f'background:{theme.BG}' in page and 'color-scheme: dark' in page
    for words in ('dnav planning — ok', 'planned route trusted', 'unknown cells', 'Status history', 'scan'):
        assert words in page, words
    assert '<wait>' not in page and '&lt;wait&gt;' in page, 'reasons must be escaped'
    assert json.loads((tmp_path / 'summary.json').read_text())['plans'] == 2


def test_a_partial_summary_still_gets_a_page(tmp_path) -> None:
    out = write_html(tmp_path, dict(reason='operator requested shutdown'))
    page = out.read_text()
    assert 'dnav planning — unknown' in page and 'no route image' in page
