#ifndef SIMOPTICON_DISCRETEPARAMETER_H
#define SIMOPTICON_DISCRETEPARAMETER_H


#include "Parameter.h"

/**
 * Implements a Parameter using discrete values.
 * The value of the Parameter is calculated as @f$ \mathit{val} = \mathit{times} \cdot \mathit{step} + \mathit{offset} @f$.
 * @ingroup parameters
 */
class DiscreteParameter : public Parameter {
private:
    /**
     * Times used in the value calculation.
     */
    int times;
    /**
     * Difference between discrete values. Used in the value calculation.
     */
    double step;
    /**
     * Offset used in the value calculation.
     */
    double offset = 0;


public:
    /**
     * Creates a DiscreteParameter with the given ParameterDefinition, step and value.
     * Checks if given value is in bounds set by the ParameterDefinition.
     * Calculates DiscreteParameter::times and DiscreteParameter::offset automatically.
     * @param def: ParameterDefinition of the Parameter.
     * @param step: Difference between discrete values.
     * @param value: Initial value of the Parameter.
     */
    DiscreteParameter(shared_ptr<ParameterDefinition> def, double step, double value);

    /**
     * Creates a DiscreteParameter with the given ParameterDefinition and step.
     * Calculates DiscreteParameter::times and DiscreteParameter::offset automatically.
     * @param def: ParameterDefinition of the Parameter.
     * @param value: Initial value of the Parameter.
     */
    DiscreteParameter(shared_ptr<ParameterDefinition> def, double step);

    /**
     * Returns the value of DiscreteParameter::times.
     * @return An integer representing the times value.
     */
    [[nodiscard]] int getTimes() const;

    /**
     * Sets the value of DiscreteParameter::times to the given value.
     * Checks if value is in bounds set by ParameterDefinition.
     * @param newTimes
     */
    void setTimes(int newTimes);

    /**
     * Returns the value of DiscreteParameter::step.
     * @return A floating point number representing the difference between discrete values.
     */
    [[nodiscard]] double getStep() const;

    /**
     * Returns the value of DiscreteParameter::offset.
     * @return A floating point number representing the offset.
     */
    [[nodiscard]] double getOffset() const;

    /**
     * Returns the current value of the DiscreteParameter as calculated by the following formula:
     * @f$ \mathit{val} = \mathit{times} \cdot \mathit{step} + \mathit{offset} @f$.
     * @return A coordinate representing the value of the ContinuousParameter.
     */
    [[nodiscard]] coordinate getVal() const override;

    /**
     * Sets the value of the DiscreteParameter to the discrete value closest to the given value by modifying DiscreteParameter::times using DiscreteParameter::setTimes.
     * @param val: Value to set the DiscreteParameter to.
     */
    void setVal(coordinate val) override;
};


#endif //SIMOPTICON_DISCRETEPARAMETER_H
