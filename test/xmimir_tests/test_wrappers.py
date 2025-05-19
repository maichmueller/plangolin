from test.fixtures import small_blocks  # noqa: F401, F403

from xmimir import *  # noqa: F401, F403
from xmimir.wrappers import atom_str_template


def test_atom_repr_matches_pymimir(small_blocks):
    space = small_blocks[0]
    for state in space:
        for atom in state.atoms():
            assert atom_str_template.render(
                predicate=atom.predicate.name, objects=atom.objects
            ) == str(atom.base)
