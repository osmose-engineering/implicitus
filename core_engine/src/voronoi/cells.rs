use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{ndarray::Array3, PyArray3};
use rayon::prelude::*;
use std::collections::HashSet;
use std::cmp::{min, max};

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 { return vec![start]; }
    let step = (end - start) / ((n-1) as f64);
    (0..n).map(|i| start + i as f64 * step).collect()
}

fn index(i: usize, j: usize, k: usize, ny: usize, nz: usize) -> usize {
    i * ny * nz + j * nz + k
}

#[pyfunction(signature = (
    points,
    bbox_min,
    bbox_max,
    resolution=None,
    wall_thickness=None
))]
pub fn construct_voronoi_cells(
    py: Python<'_>,
    points: Vec<(f64,f64,f64)>,
    bbox_min: (f64,f64,f64),
    bbox_max: (f64,f64,f64),
    resolution: Option<(usize,usize,usize)>,
    wall_thickness: Option<f64>,
) -> PyResult<(Vec<PyObject>, Vec<(usize,usize)>, Vec<Vec<usize>>)> {
    let (nx,ny,nz) = resolution.unwrap_or((32,32,32));
    let wall = wall_thickness.unwrap_or(0.0);
    let xs = linspace(bbox_min.0, bbox_max.0, nx);
    let ys = linspace(bbox_min.1, bbox_max.1, ny);
    let zs = linspace(bbox_min.2, bbox_max.2, nz);
    let nvox = nx*ny*nz;
    let npts = points.len();
    let pts_x: Vec<f64> = points.iter().map(|p| p.0).collect();
    let pts_y: Vec<f64> = points.iter().map(|p| p.1).collect();
    let pts_z: Vec<f64> = points.iter().map(|p| p.2).collect();

    let mut nearest = vec![usize::MAX; nvox];
    let mut d0 = vec![f64::INFINITY; nvox];
    let mut d1 = vec![f64::INFINITY; nvox];

    py.allow_threads(||{
        nearest.par_iter_mut()
            .zip(d0.par_iter_mut())
            .zip(d1.par_iter_mut())
            .enumerate()
            .for_each(|(idx, ((n_ref, d0_ref), d1_ref))| {
                let k = idx % nz; let j = (idx / nz) % ny; let i = idx / (ny*nz);
                let x = xs[i]; let y = ys[j]; let z = zs[k];
                let mut best0 = f64::INFINITY; let mut best1 = f64::INFINITY; let mut best_idx = 0usize;
                for s in 0..npts {
                    let dx = x - pts_x[s];
                    let dy = y - pts_y[s];
                    let dz = z - pts_z[s];
                    let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                    if dist < best0 { best1 = best0; best0 = dist; best_idx = s; }
                    else if dist < best1 { best1 = dist; }
                }
                *n_ref = best_idx; *d0_ref = best0; *d1_ref = best1;
            });
    });

    // adjacency extraction
    let edge_set: HashSet<(usize,usize)> = (0..nx).into_par_iter().map(|i|{
        let mut local = HashSet::new();
        for j in 0..ny { for k in 0..nz {
            let idx = index(i,j,k,ny,nz);
            let a = nearest[idx];
            if i+1 < nx {
                let b = nearest[index(i+1,j,k,ny,nz)];
                if a!=b { local.insert((min(a,b), max(a,b))); }
            }
            if j+1 < ny {
                let b = nearest[index(i,j+1,k,ny,nz)];
                if a!=b { local.insert((min(a,b), max(a,b))); }
            }
            if k+1 < nz {
                let b = nearest[index(i,j,k+1,ny,nz)];
                if a!=b { local.insert((min(a,b), max(a,b))); }
            }
        }}
        local
    }).reduce(|| HashSet::new(), |mut a,b| { a.extend(b); a });
    let mut edges: Vec<(usize,usize)> = edge_set.into_iter().collect();
    edges.sort();
    let mut neighbors = vec![Vec::<usize>::new(); npts];
    for (a,b) in &edges { neighbors[*a].push(*b); neighbors[*b].push(*a); }

    // build grids
    let outside = 1e9;
    let mut grids: Vec<Vec<f64>> = vec![vec![outside; nvox]; npts];
    for idx in 0..nvox { let c=nearest[idx]; let val=d1[idx]-d0[idx]-wall*0.5; grids[c][idx]=val; }

    let mut cells: Vec<PyObject> = Vec::new();
    for (ci, seed) in points.iter().enumerate() {
        let arr = PyArray3::from_owned_array_bound(
            py,
            Array3::from_shape_vec((nx, ny, nz), grids[ci].clone()).unwrap(),
        );
        let dict = PyDict::new_bound(py);
        dict.set_item("site", seed)?;
        dict.set_item("sdf", arr)?;
        dict.set_item("vertices", Vec::<(f64,f64,f64)>::new())?;
        dict.set_item("volume", 0.0)?;
        dict.set_item("neighbors", neighbors[ci].clone())?;
        cells.push(dict.into_py(py));
    }
    Ok((cells, edges, neighbors))
}

#[pyfunction(signature = (
    points,
    bbox_min,
    bbox_max,
    resolution=None,
    wall_thickness=None
))]
pub fn construct_surface_voronoi_cells(
    py: Python<'_>,
    points: Vec<(f64,f64,f64)>,
    bbox_min: (f64,f64,f64),
    bbox_max: (f64,f64,f64),
    resolution: Option<(usize,usize,usize)>,
    wall_thickness: Option<f64>,
) -> PyResult<(Vec<PyObject>, Vec<(usize,usize)>, Vec<Vec<usize>>)> {
    // For simplicity reuse volumetric logic with d_seed instead of d1-d0
    let (nx,ny,nz) = resolution.unwrap_or((32,32,32));
    let wall = wall_thickness.unwrap_or(1.0);
    let xs = linspace(bbox_min.0, bbox_max.0, nx);
    let ys = linspace(bbox_min.1, bbox_max.1, ny);
    let zs = linspace(bbox_min.2, bbox_max.2, nz);
    let nvox = nx*ny*nz;
    let npts = points.len();
    let pts_x: Vec<f64> = points.iter().map(|p| p.0).collect();
    let pts_y: Vec<f64> = points.iter().map(|p| p.1).collect();
    let pts_z: Vec<f64> = points.iter().map(|p| p.2).collect();
    let mut nearest = vec![usize::MAX; nvox];
    let mut dseed = vec![f64::INFINITY; nvox];

    py.allow_threads(||{
        nearest.par_iter_mut()
            .zip(dseed.par_iter_mut())
            .enumerate()
            .for_each(|(idx, (n_ref, d_ref))| {
                let k = idx % nz; let j = (idx / nz) % ny; let i = idx / (ny*nz);
                let x = xs[i]; let y = ys[j]; let z = zs[k];
                let mut best = f64::INFINITY; let mut best_idx = 0usize;
                for s in 0..npts {
                    let dx = x-pts_x[s]; let dy=y-pts_y[s]; let dz=z-pts_z[s];
                    let dist=(dx*dx+dy*dy+dz*dz).sqrt();
                    if dist < best { best=dist; best_idx=s; }
                }
                *n_ref = best_idx; *d_ref = best;
            });
    });

    let edge_set: HashSet<(usize,usize)> = (0..nx).into_par_iter().map(|i|{
        let mut local=HashSet::new();
        for j in 0..ny { for k in 0..nz {
            let idx=index(i,j,k,ny,nz); let a=nearest[idx];
            if i+1<nx { let b=nearest[index(i+1,j,k,ny,nz)]; if a!=b { local.insert((min(a,b),max(a,b))); }}
            if j+1<ny { let b=nearest[index(i,j+1,k,ny,nz)]; if a!=b { local.insert((min(a,b),max(a,b))); }}
            if k+1<nz { let b=nearest[index(i,j,k+1,ny,nz)]; if a!=b { local.insert((min(a,b),max(a,b))); }}
        }}
        local
    }).reduce(||HashSet::new(), |mut a,b|{a.extend(b);a});
    let mut edges: Vec<(usize,usize)>=edge_set.into_iter().collect();
    edges.sort();
    let mut neighbors=vec![Vec::<usize>::new(); npts];
    for (a,b) in &edges { neighbors[*a].push(*b); neighbors[*b].push(*a); }

    let outside=1e9; let mut grids: Vec<Vec<f64>>=vec![vec![outside;nvox];npts];
    for idx in 0..nvox { let c=nearest[idx]; let val=dseed[idx]-wall*0.5; grids[c][idx]=val; }

    let mut cells: Vec<PyObject>=Vec::new();
    for (ci, seed) in points.iter().enumerate() {
        let arr = PyArray3::from_owned_array_bound(
            py,
            Array3::from_shape_vec((nx, ny, nz), grids[ci].clone()).unwrap(),
        );
        let dict = PyDict::new_bound(py);
        dict.set_item("site", seed)?;
        dict.set_item("sdf", arr)?;
        dict.set_item("vertices", Vec::<(f64,f64,f64)>::new())?;
        dict.set_item("area", 0.0)?;
        dict.set_item("neighbors", neighbors[ci].clone())?;
        cells.push(dict.into_py(py));
    }
    Ok((cells, edges, neighbors))
}
