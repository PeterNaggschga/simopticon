#include "Optimizer.h"

#include "../controller/Controller.h"
#include "../parameters/Parameter.h"

#include <utility>

map<vector<shared_ptr<Parameter>>, functionValue>
Optimizer::requestValues(const list<vector<shared_ptr<Parameter>>> &params) {
    return controller.requestValues(params);
}

Optimizer::Optimizer(Controller &ctrl, list<shared_ptr<ParameterDefinition>> params) : controller(ctrl),
                                                                                       parameters(std::move(params)) {
}

ValueMap &Optimizer::getValueMap() const {
    return controller.getValueMap();
}
