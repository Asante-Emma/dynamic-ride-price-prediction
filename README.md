<h1><strong>Predicting Ride Prices with Machine Learning for Dynamic Pricing</strong></h1>

<h2>What Is Dynamic Pricing Strategy?</h2>
<p>A dynamic pricing approach represents a commercial tactic wherein the cost of a product or service varies in accordance with market demand. It operates under the premise that the level of demand for a particular product or service fluctuates, necessitating price modifications to align with these shifts.
Dynamic pricing manifests in various scenarios, such as airlines raising ticket prices during peak travel seasons like holidays, while hotels adjust rates upwards during events drawing significant crowds.

Again, dynamic pricing adapts to market fluctuations. For instance, when demand spikes for a product, its price may surge accordingly; conversely, prices might decrease if demand wanes. This dynamic approach is often labeled as 'demand-driven pricing' or 'market-responsive pricing'.</p>

<p>Dynamic pricing has always fascinated me. It's essentially using data science as a secret weapon to adjust prices on the fly, maximizing both revenue and customer satisfaction for businesses. Instead of static prices, companies can leverage real-time data to tailor prices to market demand, customer behavior, and competitor strategies.

Imagine I'm working with a ride-sharing company in a bustling city. Their current flat rate per kilometer just doesn't capture the dynamic nature of the transportation landscape.

Here's where data science comes in. We can gather a treasure trove of data – historical trip information, real-time demand surges, even traffic patterns and local events. By feeding this data into machine learning algorithms, we can create a dynamic pricing model.

During peak hours or when a concert lets out, the algorithm can strategically raise fares. This incentivizes more drivers to hit the road, balancing supply and demand. Conversely, during slow periods, the algorithm could lower prices to attract customers and keep drivers busy.

This project excites me because it allows me to explore the power of data science in a real-world scenario. By analyzing vast amounts of data and building a dynamic pricing model, I can directly impact a company's revenue and customer experience. It's a win-win – for the business and for the riders!</p>

<p>In this project, I'm particularly interested in the concept of dynamic pricing. It's all about finding the pricing sweet spot – maximizing revenue and profitability while keeping customers happy. Businesses can achieve this by dynamically adjusting prices based on various factors. Imagine rush hour commutes costing more than a midday ride, or concert tickets fluctuating based on demand – that's the power of dynamic pricing!

To make this work, we need a rich data playground. This means gathering historical sales data to understand past trends, purchase patterns to identify customer behavior, and market demand forecasts to anticipate future needs. Additionally, cost data is crucial to ensure healthy profit margins, and customer segmentation allows us to tailor pricing strategies to specific groups. Finally, real-time market data becomes the magic ingredient, allowing us to adjust prices on the fly based on what's happening right now.

By building a dynamic pricing model using this data, I'm looking to create a win-win situation. Businesses can optimize their revenue and customer satisfaction, while I gain valuable experience in the exciting world of data science.</p>

<h3>Dataset</h3>
<p>The dataset for this project is based on the following:</p>
<ul>
<li>historical sales data</li>
<li>customer purchase patterns</li>
<li>market demand forecasts</li>
<li>cost data</li>
<li>customer segmentation data</li>
<li>real-time market data</li>
</ul>
<a href="https://statso.io/wp-content/uploads/2023/06/dynamic_pricing.csv">Source of data</a>

<h2>Implementing the dynamic pricing strategy</h2>

<p>The development of the dynamic pricing strategy is to adjust ride costs based on real-time data insights. The goal is to fine-tune fares in response to changes in demand and supply. During busy periods when demand outstrips supply, I'll raise prices to reflect the increased demand. Conversely, during quieter times when supply exceeds demand, we'll lower prices to encourage more bookings</p>

<h3>Base Price</h3>
<ul>
    <li>Utilize the median historical cost for each vehicle type (Economy, Premium) as the base price.</li>
    <li>This reflects past pricing trends and provides a baseline for adjustments.</li>
</ul>

<h3>Demand Multiplier</h3>
<ul>
    <li>The demand multiplier represents how much the price should be adjusted based on the level of demand for rides.</li>
    <li>Calculate the demand ratio by comparing the number of riders for a specific ride to a percentile representing high demand in your data.</li>
    <li>Apply a logarithmic function (np.log1p) to create a smoother transition in the multiplier value as demand increases.</li>
    <li>Ensure a minimum impact on price with a threshold. Cap the multiplier's impact with another threshold.</li>
</ul>

<h3>Supply Multiplier</h3>
<ul>
    <li>The supply multiplier represents how much the price should be adjusted based on the level of supply of drivers available for rides.</li>
    <li>Calculate the supply ratio by comparing the available drivers in the area to a percentile representing low supply in your data.</li>
    <li>Use the logarithmic function (np.log1p) for a smoother transition in the multiplier value.</li>
    <li>Set thresholds to control the multiplier's impact on price.</li>
</ul>

<h3>Final Dynamic Price</h3>
<ul>
    <li>Multiply the base price by the maximum of the demand multiplier and its lower threshold.</li>
    <li>Multiply the result by the maximum of the supply multiplier and its lower threshold.</li>
</ul>

<h2>Understanding the Equation (Dynamic Pricing Function with Demand and Supply Multipliers)</h2>

<p>This section describes the logic behind the dynamic pricing function used in this project. The function calculates an adjusted ride price based on the historical cost, number of riders requesting the ride, and number of drivers available in the area.</p>

<h3>Demand and Supply Ratios</h3>
<p>The function considers the <strong>demand-supply ratio</strong> to determine how much to adjust the price based on current conditions. It calculates separate ratios for demand and supply using a smoother transition with the natural logarithm function:</p>

```math
\text{ratio} = \ln(1 + \text{value} / \text{threshold})
```

<ul>
    <li><strong>value:</strong> represents the current number of riders (for demand) or number of drivers (for supply).</li>
    <li><strong>threshold:<strong> Represents a percentile threshold from the historical data.</li>
</ul>

<p>These ratios are then used to calculate multipliers that influence the final price.</p>

<h3>Demand and Supply Multipliers</h3>
<p>The function applies thresholds (demand_threshold_low, demand_threshold_high, supply_threshold_low, and supply_threshold_high) to ensure the multipliers don't become too extreme. Here's the calculation of the multipliers:</p>

```math
\text{multiplier} = \max\left(\text{ratio}, \text{threshold\_low}\right) \left(\frac{\text{threshold\_high}}{\text{threshold\_low}}\right)^\text{ratio}
```

<ul>
    <li><strong>ratio:</strong> The calculated demand or supply ratio from the previous step</li>
    <li><strong>threshold_low:</strong> Lower threshold for the multiplier</li>
    <li><strong>threshold_high:<strong> Upper threshold for the multiplier.</li>
</ul>

<h3>Adjusted Ride Cost</h3>
<p>Finally, the function calculates the adjusted ride cost by multiplying the historical cost with both the demand and supply multipliers:</p>

```math
\text{adjusted\_cost} = \text{historical\_cost} \times \text{demand\_multiplier} \times \text{supply\_multiplier}
```

<h2>Model Building and Evaluation</h2>
<p>This section describes a machine learning model that was built to predict dynamic ride prices.</p>

<h3>Dynamic Pricing Function Integration</h3>
<p>The model is trained on a dataset containing historical ride information.</p>
<p>The function is applied to the ride data to compute the adjusted cost for each ride, which serves as the target variable for machine learning modeling. This function dynamically adjusts the ride cost based on the factors, demand and supply.</p>

<h3>Data Preprocessing</h3>
<p>A custom function is used to prepare the data for modeling. This function performs the following steps:</p>

<ul>
    <li><strong>Feature Identification:<strong> It identifies both numerical and categorical features based on their data types.</li>
    <li>
        Missing Value Handling:
        <ol>
            <li><strong>Numerical Features:</strong> Missing values are replaced with the median value of each numerical feature.</li>
            <li><strong>Categorical Features:</strong> Missing values are replaced with the most frequent category's value (mode) in each categorical feature.</li>
        </ol>
    </li>
    <li><strong>Outlier Treatment (Numerical Features):</strong> It identifies and handles outliers in numerical features using Interquartile Range (IQR). Outliers are values that fall outside the range of 1.5 times the IQR below the first quartile (q1) and 1.5 times the IQR above the third quartile (q3). These outliers are replaced with the closest values within the IQR range using clipping.
</li>
</ul>

<h3>Feature Selection</h3>
<p>The following features were selected as relevant for predicting ride prices:</p>

<ul>
    <li>Number of Riders</li>
    <li>Number of Drivers</li>
    <li>Location Category</li>
    <li>Customer Loyalty Status</li>
    <li>Number of Past Rides</li>
    <li>Average Ratings</li>
    <li>Time of Booking</li>
    <li>Vehicle Type</li>
    <li>Expected Ride Duration</li>
</ul>

<h3>Model Definition and Training</h3>
<p>A Random Forest Regressor is chosen for this project due to its robustness and ability to handle various data types. The model is implemented within a pipeline along with the preprocessing steps for a streamlined workflow. The model is then trained on the prepared data to learn the relationship between the features, and the actual adjusted cost</p>

<h3>Model Evaluation</h3>
<p>The model's performance is evaluated using Mean Squared Error (MSE) and R-squared. The achieved results are:</p>

<ul>
    <li><strong>Mean Squared Error (MSE):</strong> 8862.1732</li>
    <li><strong>R-squared:</strong> 0.8734</li>
</ul>
<p>A lower MSE indicates a better fit, and an R-squared value of 0.8734 suggests the model explains around 87% of the variance in the adjusted cost</p>

<h4>Correlation between the actual and predicted costs</h4>

<img width="770" alt="Screenshot 2024-03-26 at 7 34 08 PM" src="https://github.com/Asante-Emma/dynamic-ride-price-prediction/assets/122864196/e645a6a5-b3d4-4780-a01e-b7bce04808ac">

<p>The plot above illustrates a robust positive correlation between the predicted and actual costs.</p>

<h4>Feature Importance</h4>
<img width="766" alt="Screenshot 2024-03-26 at 7 45 39 PM" src="https://github.com/Asante-Emma/dynamic-ride-price-prediction/assets/122864196/cf678b0d-6717-4343-b48c-c2b2f4ae48f6">

<h2>Summary</h2>
<p><strong>Machine learning predicts ride prices!</strong> The model uses historical data (riders, drivers, location, etc.) to forecast dynamic ride costs, achieving an R-squared of 0.87. This paves the way for optimized pricing strategies in ride-hailing services.</p>

<h2>Reference</h2>
Kharwal, Aman. "Dynamic Pricing Strategy using Python." 26 June 2023. https://thecleverprogrammer.com/2023/06/26/dynamic-pricing-strategy-using-python/
