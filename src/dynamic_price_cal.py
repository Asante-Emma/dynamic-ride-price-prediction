import numpy as np

# Define percentile for high/low demand and supply
high_demand_percentile = 0.8
low_demand_percentile = 0.2

high_supply_percentile = 0.9
low_supply_percentile = 0.1

# Define thresholds for demand and supply multipliers
demand_threshold_low = 1.1 
demand_threshold_high = 1.3
supply_threshold_low = 0.9
supply_threshold_high = 0.74

def calculate_dynamic_price(historical_cost, num_riders, num_riders_series, num_drivers_series, num_drivers):
  """
  Calculates the dynamic price for a ride

  Args:
      historical_cost: The historical cost of the ride
      num_riders: The number of riders requesting the ride
      num_drivers: The number of drivers available in the area
      num_riders_series: A series of riders data
      num_drivers_series: A series of drivers data

  Returns:
      The dynamic price for the ride
  """

  # Calculate demand multiplier with smoother transition
  demand_ratio = np.log1p(num_riders / num_riders_series.quantile(high_demand_percentile))
  demand_multiplier = max(demand_ratio, demand_threshold_low) * (demand_threshold_high / demand_threshold_low) ** demand_ratio

  # Calculate supply multiplier with smoother transition
  supply_ratio = np.log1p(num_drivers_series.quantile(high_supply_percentile) / num_drivers)
  supply_multiplier = max(supply_ratio, supply_threshold_low) * (supply_threshold_high / supply_threshold_low) ** supply_ratio

  # Adjusted ride cost with base price and multipliers
  adjusted_ride_cost = historical_cost * demand_multiplier * supply_multiplier

  return adjusted_ride_cost