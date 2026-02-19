use crate::sampler::transcripts::TranscriptIndex;

/// A data structure that maps `TranscriptIndex` to values using run-length encoding.
///
/// This saves memory when many consecutive transcripts share the same value.
/// Lookup is performed using binary search on the run end indices.
#[derive(Clone, Debug)]
pub struct TranscriptRunMap<T> {
    /// Cumulative lengths of runs. The end index of each run (exclusive).
    /// The i-th run covers TranscriptIndex range [ends[i-1], ends[i]).
    ends: Vec<u32>,
    /// The value associated with each run.
    values: Vec<T>,
}

impl<T: PartialEq + Clone> TranscriptRunMap<T> {
    /// Create a new map from a sequence of run ends and their corresponding values.
    ///
    /// `ends` must be strictly increasing and `ends.len() == values.len()`.
    #[allow(dead_code)]
    pub fn new(ends: Vec<u32>, values: Vec<T>) -> Self {
        assert_eq!(ends.len(), values.len());
        TranscriptRunMap { ends, values }
    }

    pub fn empty() -> Self {
        TranscriptRunMap {
            ends: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Add the value for the next transcript.
    pub fn push(&mut self, value: T) {
        if let Some(last_value) = self.values.last() {
            if *last_value == value {
                if let Some(last_end) = self.ends.last_mut() {
                    *last_end += 1;
                    return;
                }
            }
        }

        let next_end = self.ends.last().copied().unwrap_or(0) + 1;
        self.ends.push(next_end);
        self.values.push(value);
    }

    /// Create a map from an iterator of (value, count) runs.
    #[allow(dead_code)]
    pub fn from_runs<I>(runs: I) -> Self
    where
        I: IntoIterator<Item = (T, u32)>,
    {
        let mut ends = Vec::new();
        let mut values = Vec::new();
        let mut current_end = 0u32;

        for (value, count) in runs {
            if count == 0 {
                continue;
            }
            if let Some(last_value) = values.last() {
                if *last_value == value {
                    current_end += count;
                    *ends.last_mut().unwrap() = current_end;
                    continue;
                }
            }
            current_end += count;
            ends.push(current_end);
            values.push(value);
        }

        TranscriptRunMap { ends, values }
    }

    /// Lookup the value for a given transcript index.
    ///
    /// Panics if `index` is greater than or equal to the total number of transcripts.
    pub fn get(&self, index: TranscriptIndex) -> &T {
        let run_idx = self.ends.partition_point(|&end| end <= index);
        &self.values[run_idx]
    }

    /// Returns the total number of transcripts in the map.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.ends.last().copied().unwrap_or(0) as usize
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    #[allow(dead_code)]
    pub fn shrink_to_fit(&mut self) {
        self.ends.shrink_to_fit();
        self.values.shrink_to_fit();
    }

    #[allow(dead_code)]
    pub fn iter(&self) -> TranscriptRunMapIter<T> {
        TranscriptRunMapIter {
            ends: &self.ends,
            values: &self.values,
            current_run: 0,
            current_index: 0,
        }
    }
}

pub struct TranscriptRunMapIter<'a, T> {
    ends: &'a [u32],
    values: &'a [T],
    current_run: usize,
    current_index: u32,
}

impl<'a, T> Iterator for TranscriptRunMapIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_run >= self.ends.len() {
            return None;
        }

        let value = &self.values[self.current_run];
        self.current_index += 1;

        if self.current_index >= self.ends[self.current_run] {
            self.current_run += 1;
        }

        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total_len = self.ends.last().copied().unwrap_or(0) as usize;
        let remaining = total_len.saturating_sub(self.current_index as usize);
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for TranscriptRunMapIter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::voxelcheckerboard::Voxel;

    #[test]
    fn test_transcript_run_map() {
        let voxels = vec![
            Voxel::new(1, 1, 1),
            Voxel::new(2, 2, 2),
            Voxel::new(3, 3, 3),
        ];
        let map = TranscriptRunMap::from_runs(vec![
            (voxels[0], 10),
            (voxels[1], 5),
            (voxels[2], 20),
        ]);

        assert_eq!(map.len(), 35);
        assert_eq!(*map.get(0), voxels[0]);
        assert_eq!(*map.get(9), voxels[0]);
        assert_eq!(*map.get(10), voxels[1]);
        assert_eq!(*map.get(14), voxels[1]);
        assert_eq!(*map.get(15), voxels[2]);
        assert_eq!(*map.get(34), voxels[2]);
    }

    #[test]
    fn test_transcript_run_map_push() {
        let mut map = TranscriptRunMap::new(Vec::new(), Vec::new());

        map.push(1u32);
        map.push(1u32);
        map.push(2u32);
        map.push(2u32);
        map.push(2u32);

        assert_eq!(map.len(), 5);
        assert_eq!(map.ends.len(), 2);
        assert_eq!(map.ends[0], 2);
        assert_eq!(map.ends[1], 5);
        assert_eq!(*map.get(1), 1);
        assert_eq!(*map.get(2), 2);
    }

    #[test]
    fn test_transcript_run_map_iter() {
        let map = TranscriptRunMap::from_runs(vec![
            (10u32, 2),
            (20u32, 3),
        ]);

        let values: Vec<u32> = map.iter().copied().collect();
        assert_eq!(values, vec![10, 10, 20, 20, 20]);
    }
}