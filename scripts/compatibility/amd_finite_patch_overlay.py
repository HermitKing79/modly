"""Auditable four-hook patch; never distributes or modifies a dependency header."""
HOOKS=(
('                typedef MixedIntegerProgramTraits                                                                MIP_Solver;',
 '                if (!MODLY_CLASSIFY(target_mesh, face_indices, face_supporting_planes)) { error_message_ = "insufficient certified candidate faces"; return false; }\n\n                typedef MixedIntegerProgramTraits                                                                MIP_Solver;'),
('                // Adds objective',
 '                MODLY_VETO(target_mesh, face_indices, face_supporting_planes, solver, variables);\n\n                // Adds objective'),
('                if (solver.solve()) {',
 '                MODLY_BEFORE_SOLVE();\n                if (solver.solve()) {'),
('                        const std::vector<double>& X = solver.solution();',
 '                        const std::vector<double>& X = solver.solution();\n                        if (!MODLY_SELECTION(target_mesh, face_indices, face_supporting_planes, X, total_variables)) { error_message_ = "empty optimal selection"; return false; }'),
)

def patched_header(original):
    result=original
    for before,after in HOOKS:
        if result.count(before)!=1:raise ValueError('Qualified CGAL hook anchor differs')
        result=result.replace(before,after)
    restored=result
    for before,after in reversed(HOOKS):restored=restored.replace(after,before)
    if restored!=original:raise ValueError('Unexpected modification outside hooks')
    return result
