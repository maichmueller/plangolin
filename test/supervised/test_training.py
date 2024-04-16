import pathlib
import tempfile

import pymimir as mi

from rgnet.encoding import ColorGraphEncoder
from rgnet.supervised.data import MultiInstanceSupervisedSet
from rgnet.supervised.training import Trainer
from .test_data import EmptyDataset


def test_training():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    save_dir = pathlib.Path(tempfile.gettempdir())
    save_file = save_dir / "model.pt"
    trainer = Trainer(
        train_set=MultiInstanceSupervisedSet([problem], ColorGraphEncoder(domain)),
        test_set=EmptyDataset(),
        evaluate_after_epoch=False,
        save_file=save_file,
    )
    trainer.train()

    assert save_file.exists() and save_file.is_file()


def test_evaluate():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    states = mi.StateSpace.new(
        problem, mi.GroundedSuccessorGenerator(problem)
    ).num_states()
    trainer = Trainer(
        train_set=MultiInstanceSupervisedSet([problem], ColorGraphEncoder(domain)),
        test_set=MultiInstanceSupervisedSet([problem], ColorGraphEncoder(domain)),
        evaluate_after_epoch=False,
    )
    mae, num_states = trainer.evaluate()
    assert isinstance(mae, float) and mae >= 0
    assert isinstance(num_states, int) and num_states == states
