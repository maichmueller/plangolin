import tempfile

import pymimir as mi
from torch_geometric.loader import DataLoader

from rgnet.encoding.hetero import HeteroEncoding
from rgnet.models.hetero_gnn import HeteroGNN
from rgnet.supervised.data import MultiInstanceSupervisedSet


def test_hetero_gnn():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    state_space = mi.StateSpace.new(problem, mi.GroundedSuccessorGenerator(problem))
    initial_state = state_space.get_initial_state()
    encoder = HeteroEncoding(domain, hidden_size=2)
    data = encoder.encoding_to_pyg_data(initial_state)

    model = HeteroGNN(
        hidden_size=2,
        num_layer=1,
        obj_name=encoder.obj_name,
        arity_by_pred=encoder.arity_by_pred,
    )
    out = model(data.x_dict, data.edge_index_dict)
    assert out.size() == (1,)


def test_hetero_batched():
    domain = mi.DomainParser("test/pddl_instances/blocks/domain.pddl").parse()
    problem = mi.ProblemParser("test/pddl_instances/blocks/problem.pddl").parse(domain)
    encoder = HeteroEncoding(domain, hidden_size=2)
    tmpdir: str = tempfile.mkdtemp()
    dataset = MultiInstanceSupervisedSet(
        [problem], encoder, force_reload=True, root=tmpdir
    )
    loader = DataLoader(dataset, batch_size=3)
    for batch in loader:
        model = HeteroGNN(
            hidden_size=2,
            num_layer=1,
            obj_name=encoder.obj_name,
            arity_by_pred=encoder.arity_by_pred,
        )

        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        assert out.size() == (batch.batch_size,)
