#include "ChildRect.h"

#include <utility>

ChildRect::ChildRect(dimension D, position pos, shared_ptr<HyRect> parent) : HyRect(D, pos, parent->getDepth() + 1),
                                                                             parent(std::move(parent)) {
}

array<vector<dirCoordinate>, 2> ChildRect::getSamplingVertices() {
    shared_ptr<HyRect> ptr = shared_ptr<HyRect>(parent);
    array<vector<dirCoordinate>, 2> vertices = ptr->getSamplingVertices();
    dimension psplit = ptr->getSplitDim();
    dirCoordinate length = vertices[1][psplit - 1] - vertices[0][psplit - 1];
    if (getPos() != position::RIGHT) {
        vertices[1][psplit - 1] = vertices[0][psplit - 1] + length / 3;
    }
    if (getPos() != position::LEFT) {
        vertices[0][psplit - 1] = vertices[0][psplit - 1] + 2 * length / 3;
    }
    if (getPos() == position::MIDDLE) {
        swap(vertices[0], vertices[1]);
    }
    return vertices;
}


bool ChildRect::operator==(const HyRect &rect) const {
    if (rect.getPos() != getPos()) {
        return false;
    }
    return parent == ((ChildRect &&) rect).parent;
}
