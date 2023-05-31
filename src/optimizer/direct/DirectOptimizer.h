#ifndef SIMOPTICON_DIRECTOPTIMIZER_H
#define SIMOPTICON_DIRECTOPTIMIZER_H


#include "DirectTypes.h"
#include "../Optimizer.h"
#include "StoppingCondition.h"
#include "HyRect.h"
#include "ParameterNormalizer.h"
#include "Levels.h"

#include <set>
#include <unordered_set>

class StoppingCondition;

using namespace std;

class DirectOptimizer : public Optimizer {

private:
    const dimension D;
    StoppingCondition stopCon;
    Levels level = Levels();
    ParameterNormalizer normalizer;
    map<depth, set<HyRect>> activeRects;
    unordered_set<HyRect> parentRects;

    map<vector<dirCoordinate>, functionValue> getValues(const list<vector<dirCoordinate>> &points);

    functionValue estimatedValue(map<vector<dirCoordinate>, functionValue> samples);

    list<HyRect> optimalRectangles(unsigned long m);

    void addActiveRect(HyRect rect);

    void removeActiveRect(HyRect rect);

public:
    DirectOptimizer(Controller &ctrl, list<ParameterDefinition> params, dimension D, StoppingCondition con,
                    ValueMap &map);

    void runOptimization() override;
};


#endif //SIMOPTICON_DIRECTOPTIMIZER_H
