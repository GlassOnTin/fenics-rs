//! Topological entity markers (MeshTags / MeshFunction parity with DOLFINx).
//!
//! Associates categorical tags or physical values (e.g. boundary IDs, material subdomains)
//! with topological entities (cells, facets, edges, vertices).

use std::collections::HashMap;

/// Associates markers of type T with entity indices of topological dimension DIM.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshTags<T: Clone + PartialEq> {
    /// Topological dimension of the tagged entities (e.g. 0 for vertices, DIM-1 for facets, DIM for cells)
    pub entity_dim: usize,
    /// Map from entity index to marker value
    pub values: HashMap<usize, T>,
}

impl<T: Clone + PartialEq> MeshTags<T> {
    pub fn new(entity_dim: usize) -> Self {
        Self {
            entity_dim,
            values: HashMap::new(),
        }
    }

    /// Set a tag for an entity index.
    pub fn set(&mut self, entity_idx: usize, tag: T) {
        self.values.insert(entity_idx, tag);
    }

    /// Get the tag for an entity index, if present.
    pub fn get(&self, entity_idx: usize) -> Option<&T> {
        self.values.get(&entity_idx)
    }

    /// Find all entity indices that match a given tag value.
    pub fn find(&self, target_tag: &T) -> Vec<usize> {
        let mut matching: Vec<usize> = self
            .values
            .iter()
            .filter_map(|(&idx, val)| if val == target_tag { Some(idx) } else { None })
            .collect();
        matching.sort_unstable();
        matching
    }

    /// Number of tagged entities.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Helper to create facet tags using a geometrical locator predicate.
pub fn mark_boundary_facets<F>(
    facet_midpoints: &[[f64; 2]],
    mut predicate: F,
    tag: usize,
    tags: &mut MeshTags<usize>,
) where
    F: FnMut(&[f64; 2]) -> bool,
{
    for (facet_idx, pt) in facet_midpoints.iter().enumerate() {
        if predicate(pt) {
            tags.set(facet_idx, tag);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_tags_lookup_and_filtering() {
        let mut tags = MeshTags::<usize>::new(1); // 1D facets (edges)
        tags.set(0, 10);
        tags.set(1, 20);
        tags.set(2, 10);
        tags.set(3, 30);

        assert_eq!(tags.get(1), Some(&20));
        assert_eq!(tags.get(4), None);

        let tens = tags.find(&10);
        assert_eq!(tens, vec![0, 2]);

        let thirties = tags.find(&30);
        assert_eq!(thirties, vec![3]);
    }
}
