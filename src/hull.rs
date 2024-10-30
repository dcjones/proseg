
use std::fmt::Debug;

// mod polygon_area;
// use crate::polygon_area::polygon_area;
use super::polygon_area::polygon_area;

// All this is just a mechanism to to do static dispatch on whether we are above or
// below the line in quickhull_part.
trait QuickhullSide {
    fn tricontains(u: (f32, f32), v: (f32, f32), w: (f32, f32), p: (f32, f32)) -> bool;
}

#[derive(Debug)]
struct QuickhullAbove;

impl QuickhullSide for QuickhullAbove {
    fn tricontains(u: (f32, f32), v: (f32, f32), w: (f32, f32), p: (f32, f32)) -> bool {
        !isabove(u, w, p) && !isabove(w, v, p)
    }
}

#[derive(Debug)]
struct QuickhullBelow;

impl QuickhullSide for QuickhullBelow {
    fn tricontains(u: (f32, f32), v: (f32, f32), w: (f32, f32), p: (f32, f32)) -> bool {
        isabove(u, w, p) && isabove(w, v, p)
    }
}

// pub fn compute_full_area(transcripts: &Vec<Transcript>) -> f32 {
//     let mut vertices = Vec::from_iter(transcripts.iter().map(|t| (t.x, t.y)));
//     let mut hull = Vec::new();

//     return convex_hull_area(&mut vertices, &mut hull);
// }

/// Compute the convex hull and return it's area.
pub fn convex_hull_area(vertices: &mut [(f32, f32)], hull: &mut Vec<(f32, f32)>) -> f32 {
    if vertices.len() < 3 {
        hull.clear();
        hull.extend(vertices.iter().cloned());
        return 0.0;
    }

    // find the leftmost and rightmost points
    let (l, r) = horizontal_extrema_indices(vertices);
    let (u, v) = (vertices[l], vertices[r]);

    hull.clear();
    hull.push(u);
    hull.push(v);

    // put l and r as the first two elements
    vertices.swap(0, l);
    if r == 0 {
        vertices.swap(1, l);
    } else {
        vertices.swap(1, r);
    }

    // partition into above and below the l,r line
    {
        let mut i = 1;
        let mut j = vertices.len();
        loop {
            i += 1;
            while (i < j) && isabove(u, v, vertices[i]) {
                i += 1;
            }

            j -= 1;
            while (j > i) && !isabove(u, v, vertices[j]) {
                j -= 1;
            }

            if i >= j {
                break;
            }

            vertices.swap(i, j);
        }

        quickhull_part(QuickhullAbove {}, &mut vertices[2..i], hull, u, v);
        quickhull_part(QuickhullBelow {}, &mut vertices[i..], hull, u, v);
    }

    // compute the area
    polygon_area(hull)
}

fn quickhull_part<T>(
    _side: T,
    vertices: &mut [(f32, f32)],
    hull: &mut Vec<(f32, f32)>,
    u: (f32, f32),
    v: (f32, f32),
) where
    T: QuickhullSide + Debug,
{
    // TODO: Should we be doing something with side here?

    if vertices.is_empty() {
        return;
    }

    // find furthest point from the u,v line
    let mut k = 0;
    let mut maxdist = 0.0;
    for (i, w) in vertices.iter().enumerate() {
        let dist = linedist(u, v, *w);
        if dist > maxdist {
            k = i;
            maxdist = dist;
        }
    }

    let w = vertices[k];
    hull.push(vertices[k]);
    vertices.swap(0, k);

    let mut i = 0;
    let mut j = vertices.len();
    loop {
        i += 1;
        while (i < j) && T::tricontains(u, v, w, vertices[i]) {
            i += 1;
        }

        j -= 1;
        while (j > i) && !T::tricontains(u, v, w, vertices[j]) {
            j -= 1;
        }

        if i >= j {
            break;
        }

        vertices.swap(i, j);
    }

    quickhull_part(_side, &mut vertices[i..], hull, u, v);
}

fn horizontal_extrema_indices(vertices: &[(f32, f32)]) -> (usize, usize) {
    let (mut i_min, mut i_max) = (0, 0);
    for (i, v) in vertices.iter().enumerate() {
        if v.0 < vertices[i_min].0 {
            i_min = i;
        }
        if v.0 > vertices[i_max].0 {
            i_max = i;
        }
    }

    (i_min, i_max)
}

// Compute the distance from the line uv to the point w.
fn linedist(u: (f32, f32), v: (f32, f32), w: (f32, f32)) -> f32 {
    let a = v.0 - u.0;
    let b = v.1 - u.1;
    (a * (u.1 - w.1) - b * (u.0 - w.0)).abs() / (a * a + b * b).sqrt()
}

// Test if the point w is above the line from u to v.
fn isabove(u: (f32, f32), v: (f32, f32), w: (f32, f32)) -> bool {
    (v.0 - u.0) * (w.1 - u.1) - (v.1 - u.1) * (w.0 - u.0) > 0.0
}
