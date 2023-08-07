/**
 * @file
 * In this file, the implementation of the Optimizer class is defined.
 */

#include "Optimizer.h"

#include "../controller/Controller.h"

#include <utility>

std::map<parameterCombination, functionValue>
Optimizer::requestValues(const std::list<parameterCombination> &params) {
    return controller.requestValues(params);
}

Optimizer::Optimizer(Controller &ctrl, list<shared_ptr<ParameterDefinition>> params) : controller(ctrl),
                                                                                       parameters(std::move(params)) {
}

ValueMap &Optimizer::getValueMap() const {
    return controller.getValueMap();
}

string Optimizer::getName() {
    return Status::getName();
}

string Optimizer::getStatus() {
    return Status::getStatus();
}

string Optimizer::getStatusBar() {
    return Status::getStatusBar();
}
