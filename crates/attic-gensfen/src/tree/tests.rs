//! The frontier's behaviour, driven by hand without a search behind it.

use super::*;

fn frontier(roots: &[&str], stop: u16, width: usize) -> Frontier {
    banded(roots, stop, width, 0)
}

fn banded(roots: &[&str], stop: u16, width: usize, band: i32) -> Frontier {
    let mut options = Options::for_tests();
    options.book_ply = stop;
    options.book_width = width;
    options.book_band = band;
    options.out = std::env::temp_dir().join(format!(
        "gensfen-tree-test-{}-{stop}-{width}-{band}.bin",
        std::process::id()
    ));
    Frontier::new(roots.iter().map(|s| (*s).to_string()).collect(), &options)
}

const START: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

#[test]
fn a_level_does_not_close_while_a_worker_still_holds_a_board() {
    let f = frontier(&[START, START], 9, 0);
    // Both roots are the same board, so the frontier hands out two tasks:
    // it dedups children, not the roots it was given.
    let first = f.claim().expect("a root");
    let second = f.claim().expect("the other root");
    f.finish(
        Some((0, first.sfen, first.parent)),
        vec![(1, "child-a".into(), "p".into())],
    );
    // The level is not closed yet, so the child is not handed out.
    assert_eq!(f.counts(), (1, 0, 0));
    f.finish(
        Some((0, second.sfen, second.parent)),
        vec![(2, "child-b".into(), "p".into())],
    );

    let next = f.claim().expect("a child");
    assert!(next.sfen.starts_with("child-"));
}

#[test]
fn a_transposition_is_queued_once() {
    let f = frontier(&[START], 9, 0);
    let task = f.claim().expect("the root");
    f.finish(
        Some((0, task.sfen, task.parent)),
        vec![
            (7, "same".into(), "p".into()),
            (7, "same-again".into(), "p".into()),
        ],
    );

    // Finished before the next claim: a worker still holding a board is
    // what the frontier waits for, so claiming twice without finishing
    // would block rather than fail.
    let child = f.claim().expect("the first of the pair");
    f.finish(Some((0, child.sfen, child.parent)), Vec::new());

    assert!(f.claim().is_none(), "the second shared its key");
    assert_eq!(f.counts(), (2, 0, 0), "the root and one child, not two");
}

#[test]
fn a_level_is_on_disk_before_the_level_below_it_is_searched() {
    // A book to move 32 is stopped by hand rather than by finishing, so
    // every level it did finish has to be on disk while it runs — that is
    // how a move-16 book comes out of a move-32 run. The root level is
    // written as its own search scored it, before any child has been
    // handed out, and rewritten later once there is something to back up.
    let mut options = Options::for_tests();
    options.book_ply = 9;
    options.out =
        std::env::temp_dir().join(format!("gensfen-tree-flush-{}.bin", std::process::id()));
    let level = options.out.with_extension("ply01.txt");
    let scores = options.out.with_extension("ply01.cp.txt");
    let _ = std::fs::remove_file(&level);
    let _ = std::fs::remove_file(&scores);
    let f = Frontier::new(vec![START.to_string()], &options);

    let root = f.claim().expect("the root");
    let sfen = root.sfen.clone();
    f.finish(
        Some((94, root.sfen, root.parent)),
        vec![(1, "a".into(), sfen.clone())],
    );
    // The frontier closes a level from `claim`, so this is the first
    // moment the root level could be on disk — and the child it hands back
    // has not been searched.
    let child = f.claim().expect("the child");
    assert_eq!(
        std::fs::read_to_string(&level).expect("the root level is on disk"),
        format!("sfen {sfen}\n")
    );
    assert!(
        std::fs::read_to_string(&scores)
            .expect("its scores too")
            .starts_with("94\t"),
        "as its own search scored it, not backed up from a child nobody has searched"
    );
    assert_eq!(child.sfen, "a");
}

#[test]
fn a_level_is_written_with_the_value_backed_up_from_its_children() {
    // The root's own search says +500, which is wrong: its best reply
    // leaves the opponent at +10, so one ply of lookahead values it at
    // -10. That is what the book gets, not the +500.
    let mut options = Options::for_tests();
    options.book_ply = 9;
    options.move_margin = 0;
    options.out =
        std::env::temp_dir().join(format!("gensfen-tree-backup-{}.bin", std::process::id()));
    let f = Frontier::new(vec![START.to_string()], &options);

    let root = f.claim().expect("the root");
    f.finish(
        Some((500, root.sfen.clone(), root.parent)),
        vec![
            (1, "a".into(), root.sfen.clone()),
            (2, "b".into(), root.sfen.clone()),
        ],
    );
    for (task, cp) in [(f.claim(), 10), (f.claim(), 40)] {
        let task = task.expect("a child");
        f.finish(Some((cp, task.sfen, task.parent)), Vec::new());
    }

    // The frontier closes a level from `claim`, so the book is not on disk
    // until a worker comes back for work that is no longer there.
    assert!(f.claim().is_none(), "the tree ran out of levels");

    let path = options.out.with_extension("ply01.cp.txt");
    let written =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let _ = std::fs::remove_file(&path);
    assert!(
        written.starts_with("-10\t"),
        "backed up from the best child, got {written:?}"
    );
}

#[test]
fn a_child_its_own_search_refutes_goes_down_with_its_children() {
    // Three siblings under one parent, at +10, +40 and +200 from their own
    // side's point of view — so from the parent's, +200 is the one it did
    // worst against. With a margin of 50 the cut sits at +60: the first two
    // stay and the third goes, taking its child with it.
    let mut options = Options::for_tests();
    options.book_ply = 9;
    options.move_margin = 50;
    options.out =
        std::env::temp_dir().join(format!("gensfen-tree-margin-{}.bin", std::process::id()));
    let f = Frontier::new(vec![START.to_string()], &options);

    let root = f.claim().expect("the root");
    f.finish(
        Some((0, root.sfen.clone(), root.parent)),
        vec![
            (1, "a".into(), root.sfen.clone()),
            (2, "b".into(), root.sfen.clone()),
            (3, "c".into(), root.sfen.clone()),
        ],
    );
    let score = |sfen: &str| match sfen {
        "a" => 10,
        "b" => 40,
        _ => 200,
    };
    let mut level = Vec::new();
    for _ in 0..3 {
        level.push(f.claim().expect("a child"));
    }
    for task in level {
        let cp = score(&task.sfen);
        let child = format!("under-{}", task.sfen);
        let parent = task.sfen.clone();
        f.finish(
            Some((cp, task.sfen, task.parent)),
            vec![(cp as u64 + 10, child, parent)],
        );
    }

    let mut taken = Vec::new();
    while let Some(task) = f.claim() {
        taken.push(task.sfen.clone());
        f.finish(Some((0, task.sfen, task.parent)), Vec::new());
    }
    taken.sort();
    assert_eq!(taken, vec!["under-a".to_string(), "under-b".to_string()]);
    assert_eq!(f.counts().2, 1, "one child refuted by its own search");
}

#[test]
fn the_root_level_sets_the_band_instead_of_being_cut_by_it() {
    // Four roots at +200, +210, +190 and +600. None of them can be outside
    // a band that does not exist yet, so all four stay, and their median is
    // what every later level is measured from. Taking one root's score
    // would let a single wild starting position carry the whole book.
    let f = banded(&[START, "b1", "b2", "b3"], 9, 0, 50);
    let score = |sfen: &str| match sfen {
        "b1" => 210,
        "b2" => 190,
        "b3" => 600,
        _ => 200,
    };
    let mut roots = Vec::new();
    for _ in 0..4 {
        roots.push(f.claim().expect("a root"));
    }
    for (i, task) in roots.into_iter().enumerate() {
        let parent = task.sfen.clone();
        f.finish(
            Some((score(&task.sfen), task.sfen, task.parent)),
            vec![(i as u64, format!("child-of-{parent}"), parent)],
        );
    }
    // The level closes as a worker comes back for the next one.
    assert!(f.claim().is_some(), "the children are the next level");
    assert_eq!(f.counts().1, 0, "the root level is never banded");
}

#[test]
fn the_band_is_measured_from_the_root_and_takes_the_children_too() {
    // The root's own search says +200, so +200 is the anchor: a band read
    // from zero would have thrown the root away and everything under it.
    // Children are scored from the other side to move, so one at -210 sits
    // at +210 in the root's frame and stays inside a band of 50, while one
    // at -600 is 400 away and goes, taking its own child with it.
    let f = banded(&[START], 9, 0, 50);
    let root = f.claim().expect("the root");
    f.finish(
        Some((200, root.sfen.clone(), root.parent)),
        vec![
            (1, "a".into(), root.sfen.clone()),
            (2, "b".into(), root.sfen.clone()),
            (3, "c".into(), root.sfen.clone()),
        ],
    );
    let score = |sfen: &str| match sfen {
        "a" => -210,
        "b" => -190,
        _ => -600,
    };
    let mut level = Vec::new();
    for _ in 0..3 {
        level.push(f.claim().expect("a child"));
    }
    for (i, task) in level.into_iter().enumerate() {
        let child = format!("under-{}", task.sfen);
        let parent = task.sfen.clone();
        f.finish(
            Some((score(&task.sfen), task.sfen, task.parent)),
            vec![(10 + i as u64, child, parent)],
        );
    }

    let mut taken = Vec::new();
    while let Some(task) = f.claim() {
        taken.push(task.sfen.clone());
        // Back in the root's frame at this level, so +200 is the anchor
        // itself and these stay.
        f.finish(Some((200, task.sfen, task.parent)), Vec::new());
    }
    taken.sort();
    assert_eq!(
        taken,
        vec!["under-a".to_string(), "under-b".to_string()],
        "the -600 board's child went with it"
    );
    assert_eq!(f.counts().1, 1, "one board outside the band");
}

#[test]
fn a_level_wider_than_the_width_is_cut_to_it() {
    let f = frontier(&[START], 9, 2);
    let task = f.claim().expect("the root");
    let children = (0..8)
        .map(|i| (i, format!("child-{i}"), "p".to_string()))
        .collect();
    f.finish(Some((0, task.sfen, task.parent)), children);

    let mut taken = 0;
    while let Some(task) = f.claim() {
        taken += 1;
        f.finish(Some((0, task.sfen, task.parent)), Vec::new());
    }
    assert_eq!(taken, 2, "eight children, a width of two");
}

#[test]
fn the_last_level_is_not_expanded() {
    // The root already carries move number 1, so a book that stops there
    // has one level and nothing below it.
    let f = frontier(&[START], 1, 0);
    let task = f.claim().expect("the root");
    assert!(!task.expand);
    f.finish(Some((0, task.sfen, task.parent)), Vec::new());
    assert!(f.claim().is_none());
}
