"""Necessary conditions from CGAL's actual fan constraints; no coordinate merging."""
def prune(count, fans, unsupported):
    if not 4 <= count <= 20000:
        raise ValueError('Candidate count')
    dead=set(unsupported)
    if any(type(x) is not int or not 0 <= x < count for x in dead):
        raise ValueError('Unsupported index')
    if not fans or len(fans)>200000:
        raise ValueError('Fan count')
    seen=set()
    for fan in fans:
        if not fan or any(type(x) is not int or not 0<=x<count for x in fan):
            raise ValueError('Fan index')
        # Duplicate occurrences have coefficient semantics not handled here.
        if len(set(fan))!=len(fan):raise ValueError('Repeated face in fan')
        seen.update(fan)
    if seen!=set(range(count)):raise ValueError('Incomplete face incidence')
    steps=[]
    changed=True
    while changed:
        changed=False
        for i,fan in enumerate(fans):
            live=[x for x in fan if x not in dead]
            if live and (len(fan)!=4 or len(live)<2):
                steps.append(dict(fan=i,faces_forced_zero=live,
                                  already_zero=[x for x in fan if x in dead],
                                  rule='NONFOUR_SUM_ZERO' if len(fan)!=4 else 'FOUR_FAN_SINGLETON'))
                dead.update(live);changed=True
    return dict(classification='PROVEN_NO_NONEMPTY_SELECTION' if len(dead)==count else 'INCONCLUSIVE',
                initial_unsupported=sorted(unsupported),remaining=sorted(set(range(count))-dead),
                forced_zero=len(dead),steps=steps)
