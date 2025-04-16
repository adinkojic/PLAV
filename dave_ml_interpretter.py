import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class DaveMLModel:
    def __init__(self, filepath):
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()
        self.ns = {"dml": "http://daveml.org/2010/DAVEML"}
        self.functions = self._extract_functions()
        self.breakpoints = self._extract_breakpoints()

    def _extract_breakpoints(self):
        bp_defs = self.root.findall(".//dml:breakpointDef", self.ns)
        bp_dict = {}
        for bp in bp_defs:
            bp_id = bp.attrib['bpID']
            values_elem = bp.find("dml:values", self.ns)
            if values_elem is not None:
                values = list(map(float, values_elem.text.replace(",", " ").split()))
                bp_dict[bp_id] = values
        return bp_dict

    def _extract_functions(self):
        funcs = {}
        for func in self.root.findall("dml:function", self.ns):
            name = func.attrib.get("name")
            data_elem = func.find(".//dml:dataTable", self.ns)
            bp_elems = func.findall(".//dml:bpRef", self.ns)
            dep_elem = func.find(".//dml:dependentVarRef", self.ns)
            if data_elem is None or dep_elem is None:
                continue
            data = list(map(float, data_elem.text.replace(",", " ").split()))
            bp_ids = [b.attrib['bpID'] for b in bp_elems]
            # Only keep functions where all breakpoints exist
            if all(bp in self.breakpoints for bp in bp_ids):
                funcs[dep_elem.attrib['varID']] = {
                    'name': name,
                    'bp_ids': bp_ids,
                    'data': data
                }
        return funcs

    def get_coefficient(self, var_id, **kwargs):
        if var_id not in self.functions:
            raise ValueError(f"Function for variable '{var_id}' not found.")

        func = self.functions[var_id]
        grids = [self.breakpoints[bp] for bp in func['bp_ids']]
        shape = tuple(len(g) for g in grids)

        if np.prod(shape) != len(func['data']):
            raise ValueError(f"Data size {len(func['data'])} does not match expected shape {shape} for variable '{var_id}'.")

        data_array = np.array(func['data']).reshape(shape)
        interpolator = RegularGridInterpolator(grids, data_array)

        # Construct input point from lowercase keys (assumes var names like 'alpha', 'el')
        point = [kwargs.get(bp.lower(), grids[i][0]) for i, bp in enumerate(func['bp_ids'])]
        return interpolator(point)

#Example usage:
model = DaveMLModel("F16_package/F16_S119_source/F16_aero.dml")
cl = model.get_coefficient("czt", alpha=10)  # Remember CZ = -CL
cd = model.get_coefficient("cxt", alpha=10, el=0)
cm = model.get_coefficient("cmt", alpha=10, el=0)
print(-cl, cd, cm)



