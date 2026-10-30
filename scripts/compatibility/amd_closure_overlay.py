"""Stage A private adapter edits; qualified dependency files are never modified."""
OLD_HOOK='MODLY_CLASSIFY(target_mesh, face_indices, face_supporting_planes)'
NEW_HOOK='phase16_capture(target_mesh, face_indices, face_supporting_planes, adjacency)'

def capture_header(original):
    if original.count(OLD_HOOK)!=1 or NEW_HOOK in original:raise ValueError('Capture anchor differs')
    result=original.replace(OLD_HOOK,NEW_HOOK)
    if result.replace(NEW_HOOK,OLD_HOOK)!=original:raise ValueError('Unexpected header edit')
    return result

def common_source(original):
    anchor='int main(int argc,char** argv){'
    if original.count(anchor)!=1:raise ValueError('Qualified main anchor differs')
    return original.replace(anchor,'int phase15_uninvoked_main(int argc,char** argv){')

def diagnostic_header(original):
    result=capture_header(original)
    hooks=(('MODLY_VETO(target_mesh, face_indices, face_supporting_planes, solver, variables)',
            'phase16_veto(target_mesh, face_indices, face_supporting_planes, solver, variables)'),
           ('MODLY_BEFORE_SOLVE()', 'phase16_before_solve()'),
           ('MODLY_SELECTION(target_mesh, face_indices, face_supporting_planes, X, total_variables)',
            'phase16_selection(target_mesh, face_indices, face_supporting_planes, X, total_variables)'))
    for before,after in hooks:
        if result.count(before)!=1:raise ValueError('Diagnostic hook anchor differs')
        result=result.replace(before,after)
    restored=result
    for before,after in reversed(hooks):restored=restored.replace(after,before)
    if restored!=capture_header(original):raise ValueError('Unexpected diagnostic edits')
    return result

def certificate_core(original):
    start=original.index('enum State {')
    end=original.index('template<class S>void modly_observe_planes(S& set){')
    core=original[start:end]
    unused='static std::vector<PNI> input_points;\n'
    cap='static constexpr size_t QUERY_LIMIT=2000000;'
    if core.count(unused)!=1 or core.count(cap)!=1:raise ValueError('Classifier extraction differs')
    return core.replace(unused,'').replace(cap,'static constexpr size_t QUERY_LIMIT=32000000;')

def refine_core(original):
    core=certificate_core(original)
    anchor='if(depth>=DEPTH_LIMIT)return r;'
    if core.count(anchor)!=1 or core.count('DEPTH_LIMIT=20;')!=1:raise ValueError('Depth instrumentation anchor differs')
    return core.replace('DEPTH_LIMIT=20;','DEPTH_LIMIT=28;').replace(anchor,
        'if(depth>=DEPTH_LIMIT){++terminal_cells;terminal_radius=std::max(terminal_radius,radius);terminal_slack=std::min(terminal_slack,EPS-MARGIN-r.distance);return r;}')
