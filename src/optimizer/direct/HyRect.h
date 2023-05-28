#ifndef SIMOPTICON_HYRECT_H
#define SIMOPTICON_HYRECT_H

#include <optional>
#include <array>
#include <vector>

using namespace std;

enum class position : char {
    LEFT = 0, MIDDLE = 1, RIGHT = 2, BASE = -1
};

class HyRect {

private:
    const unsigned char D;
    unsigned int depth; // TODO: entfernen?
    position pos;
    unsigned char split = 0;
    HyRect *parent;

public:
    HyRect(unsigned char D, position pos, HyRect *parent);

    array<HyRect *, 3> divide(unsigned char dimension);

    array<vector<double>, 2> getSamplingVertices();

    [[nodiscard]] unsigned int getDepth() const;

    [[nodiscard]] position getPos() const;

    [[nodiscard]] unsigned char getDim() const;
};


#endif //SIMOPTICON_HYRECT_H
