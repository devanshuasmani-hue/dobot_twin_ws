"""COLLADA (.dae) loader for verification/rendering: triangles in the mesh's own
frame (scene-graph transforms applied, unit scale applied) plus per-triangle colours.
Pure numpy + stdlib."""
import functools
import xml.etree.ElementTree as ET
import numpy as np


def _t(tag):
    return tag.split('}')[-1]


def _kids(el, name):
    return [c for c in el if _t(c.tag) == name]


def _node_matrix(n):
    M = np.eye(4)
    for c in n:
        t = _t(c.tag)
        if t == 'matrix':
            M = M @ np.array(c.text.split(), float).reshape(4, 4)
        elif t == 'translate':
            T = np.eye(4)
            T[:3, 3] = [float(x) for x in c.text.split()]
            M = M @ T
        elif t == 'scale':
            M = M @ np.diag([float(x) for x in c.text.split()] + [1.0])
        elif t == 'rotate':
            x, y, z, a = (float(v) for v in c.text.split())
            a = np.radians(a)
            v = np.array([x, y, z])
            v /= np.linalg.norm(v)
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Rm = np.eye(4)
            Rm[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
            M = M @ Rm
    return M


@functools.lru_cache(maxsize=None)
def load_dae(path):
    """Returns dict(tris=(N,3,3) float, colors=(N,3) float 0..1, unit=float, up=str)."""
    root = ET.parse(path).getroot()
    unit, up = 1.0, 'Y_UP'
    for e in root.iter():
        if _t(e.tag) == 'unit' and e.get('meter'):
            unit = float(e.get('meter'))
        elif _t(e.tag) == 'up_axis':
            up = e.text.strip()
    # effect colours
    eff_col = {}
    for e in root.iter():
        if _t(e.tag) == 'effect':
            col = None
            for d in e.iter():
                if _t(d.tag) in ('diffuse',):
                    c = [x for x in d if _t(x.tag) == 'color']
                    if c:
                        col = [float(v) for v in c[0].text.split()][:3]
            eff_col[e.get('id')] = col or [0.7, 0.7, 0.7]
    mat_col = {}
    for e in root.iter():
        if _t(e.tag) == 'material':
            ie = _kids(e, 'instance_effect')
            if ie:
                mat_col[e.get('id')] = eff_col.get(ie[0].get('url')[1:], [0.7, 0.7, 0.7])
    # geometries -> list of (triangle vertex array, material symbol)
    geoms = {}
    for g in root.iter():
        if _t(g.tag) != 'geometry':
            continue
        mesh = _kids(g, 'mesh')[0]
        src = {}
        for s in _kids(mesh, 'source'):
            fa = _kids(s, 'float_array')
            if fa and fa[0].text:
                src[s.get('id')] = np.array(fa[0].text.split(), float).reshape(-1, 3)
        vmap = {}
        for v in _kids(mesh, 'vertices'):
            for i in _kids(v, 'input'):
                if i.get('semantic') == 'POSITION':
                    vmap[v.get('id')] = i.get('source')[1:]
        prims = []
        for pr in mesh:
            kind = _t(pr.tag)
            if kind not in ('triangles', 'polylist', 'polygons'):
                continue
            inputs = _kids(pr, 'input')
            stride = max(int(i.get('offset')) for i in inputs) + 1
            vin = [i for i in inputs if i.get('semantic') == 'VERTEX'][0]
            off = int(vin.get('offset'))
            pos = src[vmap[vin.get('source')[1:]]]
            tris = []
            if kind == 'triangles':
                idx = np.array(' '.join(p.text for p in _kids(pr, 'p') if p.text).split(), int)
                vi = idx[off::stride]
                tris = pos[vi].reshape(-1, 3, 3)
            elif kind == 'polylist':
                vc = np.array(_kids(pr, 'vcount')[0].text.split(), int)
                idx = np.array(_kids(pr, 'p')[0].text.split(), int)
                vi = idx[off::stride]
                o = 0
                out = []
                for n in vc:
                    poly = vi[o:o + n]
                    o += n
                    for k in range(1, n - 1):
                        out.append(pos[[poly[0], poly[k], poly[k + 1]]])
                tris = np.array(out).reshape(-1, 3, 3)
            else:  # polygons
                out = []
                for p in _kids(pr, 'p'):
                    vi = np.array(p.text.split(), int)[off::stride]
                    for k in range(1, len(vi) - 1):
                        out.append(pos[[vi[0], vi[k], vi[k + 1]]])
                tris = np.array(out).reshape(-1, 3, 3)
            prims.append((tris, pr.get('material')))
        geoms[g.get('id')] = prims
    # scene graph
    scene = [s for s in root.iter() if _t(s.tag) == 'visual_scene'][0]
    libnodes = {n.get('id'): n for ln in root.iter() if _t(ln.tag) == 'library_nodes'
                for n in _kids(ln, 'node')}
    all_tris, all_cols = [], []

    def walk(n, M):
        M = M @ _node_matrix(n)
        for c in n:
            t = _t(c.tag)
            if t == 'instance_geometry':
                symmap = {}
                for im in c.iter():
                    if _t(im.tag) == 'instance_material':
                        symmap[im.get('symbol')] = im.get('target')[1:]
                for tris, sym in geoms[c.get('url')[1:]]:
                    if not len(tris):
                        continue
                    h = np.concatenate([tris.reshape(-1, 3), np.ones((tris.shape[0] * 3, 1))], 1) @ M.T
                    all_tris.append(h[:, :3].reshape(-1, 3, 3))
                    col = mat_col.get(symmap.get(sym), [0.7, 0.7, 0.7])
                    all_cols.append(np.tile(col, (len(tris), 1)))
            elif t == 'instance_node':
                ln = libnodes.get(c.get('url')[1:])
                if ln is not None:
                    walk(ln, M)
            elif t == 'node':
                walk(c, M)

    for n in _kids(scene, 'node'):
        walk(n, np.eye(4))
    tris = np.concatenate(all_tris) * unit
    return dict(tris=tris, colors=np.concatenate(all_cols), unit=unit, up=up)
