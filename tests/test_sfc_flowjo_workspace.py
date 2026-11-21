"""
test_sfc_flowjo_workspace.py

Tests for read_flowjo_gates() and apply_flowjo_gates().

A minimal FlowJo workspace is written by hand: one sample, a P1 triangle on log
FSC-A / log SSC-A axes, and a DR+ rectangle nested in P1 whose SSC-A lower bound
is 1 (the log-axis floor). The triangle is chosen so that drawing its long edge on
the displayed (log) axes or on raw values gives different answers for one event:
the test checks that the log display is used, as FlowJo does, and that values
<= 1 count as sitting on the floor of a log axis.
"""

import numpy as np
import pytest

from mito_marker.sfc.fcs_io import read_fcs_events
from mito_marker.sfc.flowjo_workspace import apply_flowjo_gates, read_flowjo_gates

WORKSPACE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Workspace version="20.0" xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating"
 xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
 xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">
 <SampleList>
  <Sample>
   <DataSet uri="file:/C:/data/tube%201.fcs" sampleID="1"/>
   <Transformations>
    <transforms:log transforms:offset="1" transforms:decades="6.6"><data-type:parameter data-type:name="FSC-A"/></transforms:log>
    <transforms:log transforms:offset="1" transforms:decades="6.6"><data-type:parameter data-type:name="SSC-A"/></transforms:log>
    <transforms:biex transforms:width="-250"><data-type:parameter data-type:name="R2-A"/></transforms:biex>
   </Transformations>
   <SampleNode name="tube 1.fcs" count="{root_count}">
    <Subpopulations>
     <Population name="P1" count="{p1_count}">
      <Gate gating:id="ID1">
       <gating:PolygonGate>
        <gating:dimension><data-type:fcs-dimension data-type:name="FSC-A"/></gating:dimension>
        <gating:dimension><data-type:fcs-dimension data-type:name="SSC-A"/></gating:dimension>
        <gating:vertex><gating:coordinate data-type:value="10"/><gating:coordinate data-type:value="10"/></gating:vertex>
        <gating:vertex><gating:coordinate data-type:value="100000"/><gating:coordinate data-type:value="10"/></gating:vertex>
        <gating:vertex><gating:coordinate data-type:value="10"/><gating:coordinate data-type:value="100000"/></gating:vertex>
       </gating:PolygonGate>
      </Gate>
      <Subpopulations>
       <Population name="DR+" count="{dr_count}">
        <Gate gating:id="ID2">
         <gating:RectangleGate>
          <gating:dimension gating:min="1000"><data-type:fcs-dimension data-type:name="Comp-R2-A"/></gating:dimension>
          <gating:dimension gating:min="1.0" gating:max="1000000"><data-type:fcs-dimension data-type:name="SSC-A"/></gating:dimension>
         </gating:RectangleGate>
        </Gate>
       </Population>
      </Subpopulations>
     </Population>
    </Subpopulations>
   </SampleNode>
  </Sample>
 </SampleList>
</Workspace>
"""


def build_sample(write_synthetic_fcs, folder):
    # Events: (FSC-A, SSC-A, R2-A)
    events = np.array([
        [100.0, 100.0, 5000.0],     # inside P1, DR+
        [100.0, 100.0, 10.0],       # inside P1, not DR+
        [30000.0, 30000.0, 5000.0],  # inside the triangle drawn on raw values, outside on log axes
        [-5.0, 100.0, 5000.0],      # FSC-A <= 1: on the log floor, outside the triangle (starts at 10)
        [1000.0, 50.0, 5000.0],     # inside P1, DR+
    ])
    write_synthetic_fcs("tube 1.fcs", {"FSC-A": events[:, 0], "SSC-A": events[:, 1], "R2-A": events[:, 2],
                                       "Time": np.arange(1, 6, dtype=float)}, folder=folder)


def write_workspace(tmp_path, root_count=5, p1_count=3, dr_count=2) -> str:
    path = tmp_path / "gated.wsp"
    path.write_text(WORKSPACE_TEMPLATE.format(root_count=root_count, p1_count=p1_count, dr_count=dr_count))
    return str(path)


def test_read_flowjo_gates_lists_nested_populations(tmp_path):
    gates = read_flowjo_gates(write_workspace(tmp_path)).set_index("population_path")
    assert list(gates.index) == ["/P1", "/P1/DR+"]
    assert gates.loc["/P1", "gate_type"] == "polygon" and len(gates.loc["/P1", "vertices"]) == 3
    assert gates.loc["/P1/DR+", "parent_path"] == "/P1"
    assert gates.loc["/P1/DR+", "bounds"][0] == (1000.0, np.inf)
    assert gates.loc["/P1", "sample_file"] == "tube 1.fcs"
    assert gates.loc["/P1", "axis_scales"]["FSC-A"] == "log"


def test_apply_flowjo_gates_reproduces_flowjo_counts_and_exports(write_synthetic_fcs, tmp_path):
    build_sample(write_synthetic_fcs, tmp_path / "fcs")
    result = apply_flowjo_gates(write_workspace(tmp_path), str(tmp_path / "fcs"),
                                populations=["P1", "DR+"], output_directory_path=str(tmp_path / "exports")).set_index("population_path")
    assert result.loc["/P1", "python_count"] == 3       # the 30000/30000 event is outside on log axes
    assert result.loc["/P1/DR+", "python_count"] == 2
    assert (result.difference == 0).all()
    assert not result.approximate.any()
    exported = read_fcs_events(str(tmp_path / "exports" / "tube 1_DR+.fcs"))
    assert exported.n_events == 2
    with pytest.raises(FileExistsError):
        apply_flowjo_gates(write_workspace(tmp_path), str(tmp_path / "fcs"), populations=["DR+"],
                           output_directory_path=str(tmp_path / "exports"))


def test_rectangle_lower_bound_at_one_includes_the_log_floor(write_synthetic_fcs, tmp_path):
    folder = tmp_path / "fcs"
    events = np.array([[100.0, -3.0, 5000.0], [100.0, 100.0, 5000.0]])  # SSC-A -3 sits on the log floor (1)
    write_synthetic_fcs("tube 1.fcs", {"FSC-A": events[:, 0], "SSC-A": events[:, 1], "R2-A": events[:, 2],
                                       "Time": np.array([1.0, 2.0])}, folder=folder)
    result = apply_flowjo_gates(write_workspace(tmp_path, root_count=2, p1_count=1, dr_count=1), str(folder), populations=["/P1/DR+"])
    # P1 excludes the floor event (polygon starts at 10), so DR+ keeps 1; the rectangle alone keeps both.
    assert result.python_count.iloc[0] == 1


def test_missing_workspace_and_missing_fcs(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_flowjo_gates(str(tmp_path / "none.wsp"))
    result = apply_flowjo_gates(write_workspace(tmp_path), str(tmp_path))
    assert result.status.iloc[0] == "fcs file not found"
