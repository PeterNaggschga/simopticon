#ifndef SIMOPTICON_PARAMETER_H
#define SIMOPTICON_PARAMETER_H


#include "../Types.h"

#include <string>
#include <memory>

class ParameterDefinition;

using namespace std;

/**
 * A class acting as the container of the value of a parameter defined by a ParameterDefinition.
 */
class Parameter {
private:
    /**
     * Reference to the defining ParameterDefinition.
     */
    shared_ptr<ParameterDefinition> definition;

public:
    /**
     * Creates a Parameter with the given ParameterDefinition.
     * @param def: Definition of properties of the Parameter.
     */
    explicit Parameter(shared_ptr<ParameterDefinition> def);

    virtual ~Parameter() = default;

    /**
     * Returns the minimum value of the Parameter stored in ParameterDefinition::min of Parameter::definition.
     * @return A coordinate representing the minimum value.
     */
    [[nodiscard]] coordinate getMin() const;

    /**
     * Returns the maximum value of the Parameter stored in ParameterDefinition::max of Parameter::definition.
     * @return A coordinate representing the maximum value.
     */
    [[nodiscard]] coordinate getMax() const;

    /**
     * Returns the unit string of the Parameter stored in ParameterDefinition::unit of Parameter::definition.
     * @return A string reference containing the unit.
     */
    [[nodiscard]] const string &getUnit() const;

    /**
     * Returns the configuration string of the Parameter stored in ParameterDefinition::config of Parameter::definition.
     * @return A string reference containing the configuration.
     */
    [[nodiscard]] const string &getConfig() const;

    /**
     * Sets the value of the Parameter to the given value.
     * @param val: Value to set the Parameter to.
     */
    virtual void setVal(coordinate val) = 0;

    bool operator==(const Parameter &rhs) const;

    bool operator!=(const Parameter &rhs) const;

    bool operator<(const Parameter &rhs) const;

    bool operator>(const Parameter &rhs) const;

    bool operator<=(const Parameter &rhs) const;

    bool operator>=(const Parameter &rhs) const;
};

#endif //SIMOPTICON_PARAMETER_H
