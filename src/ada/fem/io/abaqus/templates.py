main_inp_str = r"""*Heading
** Generated by:
** Assembly For Design and Analysis (ADA) FEM IO (Abaqus)
*Preprint, echo=NO, model=NO, history=NO, contact=NO
{constr_ctrl}
**
** -------------------------------------------------------
** PARTS
** -------------------------------------------------------
{part_str}
** -------------------------------------------------------
** ASSEMBLY
** -------------------------------------------------------
*Assembly, name=Assembly
**
{instance_str}
**
*INCLUDE,INPUT=core_input_files\connectors.inp
*INCLUDE,INPUT=core_input_files\constraints.inp
*INCLUDE,INPUT=core_input_files\assembly_data.inp
**
*End Assembly
** -------------------------------------------------------
** Connector Sections and Amplitudes
** -------------------------------------------------------
{ampl_str}
{consec_str}
**
** -------------------------------------------------------
** INTERACTION PROPERTIES
** -------------------------------------------------------
{int_prop_str}
**
** -------------------------------------------------------
** MATERIALS
** -------------------------------------------------------
{mat_str}
**
** -------------------------------------------------------
** BOUNDARY CONDITIONS
** -------------------------------------------------------
{fix_str}
**
** -------------------------------------------------------
** INTERACTIONS
** -------------------------------------------------------
{interact_str}
**
** -------------------------------------------------------
** STEPS
** -------------------------------------------------------
{step_str}"""

__all__ = ["main_inp_str"]
