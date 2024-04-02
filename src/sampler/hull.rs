use super::transcripts::{CellIndex, Transcript, BACKGROUND_CELL};

use std::cmp::Ordering;
use std::fmt::Debug;

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

pub fn compute_cell_areas(
    ncells: usize,
    transcripts: &[Transcript],
    cell_assignments: &[CellIndex],
) -> Vec<f32> {
    let mut vertices: Vec<Vec<(f32, f32)>> = vec![Vec::new(); ncells];
    for (&c, &t) in cell_assignments.iter().zip(transcripts.iter()) {
        if c != BACKGROUND_CELL {
            vertices[c as usize].push((t.x, t.y));
        }
    }

    let mut hull = Vec::new();
    let areas = vertices
        .iter_mut()
        .map(|vs| convex_hull_area(vs, &mut hull))
        .collect();

    areas
}

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

pub fn polygon_area(vertices: &mut [(f32, f32)]) -> f32 {
    let c = center(vertices);
    vertices.sort_unstable_by(|a, b| clockwise_cmp(c, *a, *b));

    let mut area = 0.0;

    for (i, u) in vertices.iter().enumerate() {
        let j = (i + 1) % vertices.len();
        let v = vertices[j];

        // triangle formula.
        // area += u.0 * v.1 - v.0 * u.1;

        // trapezoid formula (this is more numerically stable with large coordinates)
        area += (v.0 + u.0) * (v.1 - u.1);
    }
    area = area.abs() / 2.0;

    area
}

fn center(vertices: &[(f32, f32)]) -> (f32, f32) {
    let mut x = 0.0;
    let mut y = 0.0;
    for v in vertices {
        x += v.0;
        y += v.1;
    }
    x /= vertices.len() as f32;
    y /= vertices.len() as f32;
    (x, y)
}

fn clockwise_cmp(c: (f32, f32), a: (f32, f32), b: (f32, f32)) -> Ordering {
    // From: https://stackoverflow.com/a/6989383
    if a.0 - c.0 >= 0.0 && b.0 - c.0 < 0.0 {
        return Ordering::Less;
    } else if a.0 - c.0 < 0.0 && b.0 - c.0 >= 0.0 {
        return Ordering::Greater;
    } else if a.0 - c.0 == 0.0 && b.0 - c.0 == 0.0 {
        if a.1 - c.1 >= 0.0 || b.1 - c.1 >= 0.0 {
            return a.1.partial_cmp(&b.1).unwrap();
        } else {
            return b.1.partial_cmp(&a.1).unwrap();
        }
    }

    // compute the cross product of vectors (c -> a) x (c -> b)
    let det = (a.0 - c.0) * (b.1 - c.1) - (b.0 - c.0) * (a.1 - c.1);

    if det < 0.0 {
        Ordering::Less
    } else if det > 0.0 {
        Ordering::Greater
    } else {
        // points a and b are on the same line from the c
        // check which point is closer to the c
        let d1 = (a.0 - c.0) * (a.0 - c.0) + (a.1 - c.1) * (a.1 - c.1);
        let d2 = (b.0 - c.0) * (b.0 - c.0) + (b.1 - c.1) * (b.1 - c.1);
        if d1 >= d2 {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
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
