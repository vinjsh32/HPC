/**
 * @file statistical_analysis.hpp
 * @brief Advanced statistical analysis for performance benchmarking
 * @version 2.0
 * @date 2024
 * 
 * This module provides comprehensive statistical analysis capabilities for
 * performance benchmarking including:
 * - Confidence intervals and hypothesis testing
 * - Outlier detection and data cleaning
 * - Performance variance analysis
 * - Trend analysis and regression
 * - Multi-variate performance modeling
 * 
 * @author HPC Team
 * @copyright 2024 High Performance Computing Laboratory
 */

#pragma once

#include <cstddef>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Statistical summary for performance metrics
 */
typedef struct {
    double mean;
    double median;
    double std_deviation;
    double variance;
    double min_value;
    double max_value;
    
    // Robust statistics
    double trimmed_mean;        // 10% trimmed mean
    double interquartile_range; // IQR (Q3 - Q1)
    double mad;                 // Median absolute deviation
    
    // Distribution characteristics
    double skewness;
    double kurtosis;
    int is_normal_distribution; // 1 if passes normality test
    
    // Confidence intervals (95%)
    double ci_lower;
    double ci_upper;
    
    // Sample information
    int sample_size;
    int outliers_detected;
    double outlier_percentage;
    
} StatisticalSummary;

/**
 * @brief Performance comparison results
 */
typedef struct {
    char baseline_name[64];
    char comparison_name[64];
    
    // Effect size measures
    double speedup_ratio;
    double effect_size_cohens_d;
    double effect_size_glass_delta;
    
    // Statistical significance
    double t_statistic;
    double p_value;
    double degrees_of_freedom;
    int is_significant;         // 1 if p < 0.05
    
    // Confidence intervals for difference
    double difference_mean;
    double difference_ci_lower;
    double difference_ci_upper;
    
    // Power analysis
    double statistical_power;
    int recommended_sample_size;
    
} PerformanceComparison;

/**
 * @brief Regression analysis results
 */
typedef struct {
    double slope;
    double intercept;
    double r_squared;
    double correlation_coefficient;
    
    // Regression diagnostics
    double residual_std_error;
    double f_statistic;
    double p_value_regression;
    
    // Confidence intervals for parameters
    double slope_ci_lower;
    double slope_ci_upper;
    double intercept_ci_lower;
    double intercept_ci_upper;
    
    // Model validation
    int model_is_significant;
    double mean_absolute_error;
    double root_mean_square_error;
    
} RegressionAnalysis;

/**
 * @brief Multi-variate performance model
 */
typedef struct {
    int num_predictors;
    char predictor_names[10][32];
    double coefficients[10];
    double coefficient_p_values[10];
    
    double multiple_r_squared;
    double adjusted_r_squared;
    double f_statistic;
    double model_p_value;
    
    // Residual analysis
    double residual_standard_error;
    int residuals_are_normal;
    int residuals_are_homoscedastic;
    
} MultiVariateModel;

/**
 * @brief Time series analysis for performance trends
 */
typedef struct {
    int num_points;
    double* timestamps;
    double* values;
    
    // Trend analysis
    double trend_slope;
    double trend_p_value;
    int has_significant_trend;
    
    // Seasonality detection
    int has_seasonality;
    double seasonal_period;
    double seasonal_amplitude;
    
    // Change point detection
    int num_change_points;
    double change_points[10];
    double change_magnitudes[10];
    
    // Forecasting
    double next_prediction;
    double prediction_ci_lower;
    double prediction_ci_upper;
    
} TimeSeriesAnalysis;

// =====================================================
// Basic Statistical Functions
// =====================================================

/**
 * @brief Calculate comprehensive statistical summary
 * @param data Array of performance measurements
 * @param size Number of measurements
 * @param summary Output statistical summary
 * @return 0 on success, -1 on failure
 */
int stats_calculate_summary(const double* data, int size, StatisticalSummary* summary);

/**
 * @brief Calculate confidence interval for mean
 * @param data Array of measurements
 * @param size Number of measurements
 * @param confidence_level Confidence level (e.g., 0.95 for 95%)
 * @param lower_bound Output: lower bound of CI
 * @param upper_bound Output: upper bound of CI
 * @return 0 on success, -1 on failure
 */
int stats_confidence_interval(const double* data, int size, double confidence_level,
                             double* lower_bound, double* upper_bound);

/**
 * @brief Detect outliers using multiple methods
 * @param data Array of measurements
 * @param size Number of measurements
 * @param outlier_indices Output: indices of detected outliers
 * @param max_outliers Maximum number of outliers to report
 * @return Number of outliers detected
 */
int stats_detect_outliers(const double* data, int size, int* outlier_indices, int max_outliers);

/**
 * @brief Remove outliers and create cleaned dataset
 * @param data Input array of measurements
 * @param size Number of measurements
 * @param cleaned_data Output: cleaned array (must be pre-allocated)
 * @param cleaned_size Output: size of cleaned array
 * @param outlier_threshold Z-score threshold for outlier detection (e.g., 2.5)
 * @return 0 on success, -1 on failure
 */
int stats_remove_outliers(const double* data, int size, double* cleaned_data, 
                         int* cleaned_size, double outlier_threshold);

// =====================================================
// Hypothesis Testing
// =====================================================

/**
 * @brief Perform two-sample t-test comparing performance
 * @param baseline_data Array of baseline measurements
 * @param baseline_size Number of baseline measurements
 * @param comparison_data Array of comparison measurements
 * @param comparison_size Number of comparison measurements
 * @param result Output: comparison results
 * @return 0 on success, -1 on failure
 */
int stats_compare_performance(const double* baseline_data, int baseline_size,
                             const double* comparison_data, int comparison_size,
                             PerformanceComparison* result);

/**
 * @brief Test for normality using Shapiro-Wilk test
 * @param data Array of measurements
 * @param size Number of measurements
 * @param p_value Output: p-value of normality test
 * @return 1 if normal distribution, 0 if not normal, -1 on error
 */
int stats_test_normality(const double* data, int size, double* p_value);

/**
 * @brief Perform Mann-Whitney U test (non-parametric comparison)
 * @param group1_data First group measurements
 * @param group1_size Size of first group
 * @param group2_data Second group measurements  
 * @param group2_size Size of second group
 * @param u_statistic Output: U statistic
 * @param p_value Output: p-value
 * @return 0 on success, -1 on failure
 */
int stats_mann_whitney_test(const double* group1_data, int group1_size,
                           const double* group2_data, int group2_size,
                           double* u_statistic, double* p_value);

// =====================================================
// Regression and Correlation Analysis
// =====================================================

/**
 * @brief Perform simple linear regression
 * @param x_values Independent variable (e.g., problem size)
 * @param y_values Dependent variable (e.g., execution time)
 * @param size Number of data points
 * @param regression Output: regression analysis results
 * @return 0 on success, -1 on failure
 */
int stats_linear_regression(const double* x_values, const double* y_values, int size,
                           RegressionAnalysis* regression);

/**
 * @brief Perform polynomial regression (quadratic or cubic)
 * @param x_values Independent variable
 * @param y_values Dependent variable
 * @param size Number of data points
 * @param degree Polynomial degree (2 for quadratic, 3 for cubic)
 * @param coefficients Output: polynomial coefficients
 * @param r_squared Output: R-squared value
 * @return 0 on success, -1 on failure
 */
int stats_polynomial_regression(const double* x_values, const double* y_values, int size,
                              int degree, double* coefficients, double* r_squared);

/**
 * @brief Calculate correlation matrix for multiple variables
 * @param data Matrix of measurements (variables in columns)
 * @param num_samples Number of samples (rows)
 * @param num_variables Number of variables (columns)
 * @param correlation_matrix Output: correlation matrix
 * @return 0 on success, -1 on failure
 */
int stats_correlation_matrix(const double** data, int num_samples, int num_variables,
                            double** correlation_matrix);

// =====================================================
// Multi-variate Analysis
// =====================================================

/**
 * @brief Build multi-variate performance model
 * @param predictors Matrix of predictor variables
 * @param response Array of response variable (performance)
 * @param num_samples Number of samples
 * @param num_predictors Number of predictor variables
 * @param predictor_names Names of predictor variables
 * @param model Output: multi-variate model
 * @return 0 on success, -1 on failure
 */
int stats_multivariate_model(const double** predictors, const double* response,
                            int num_samples, int num_predictors, 
                            const char** predictor_names, MultiVariateModel* model);

/**
 * @brief Perform principal component analysis
 * @param data Input data matrix
 * @param num_samples Number of samples
 * @param num_variables Number of variables
 * @param components Output: principal components
 * @param eigenvalues Output: eigenvalues
 * @param explained_variance Output: explained variance ratios
 * @return Number of significant components
 */
int stats_principal_components(const double** data, int num_samples, int num_variables,
                              double** components, double* eigenvalues, 
                              double* explained_variance);

// =====================================================
// Time Series and Trend Analysis
// =====================================================

/**
 * @brief Analyze performance trends over time
 * @param timestamps Array of timestamps
 * @param performance_values Array of performance measurements
 * @param size Number of measurements
 * @param analysis Output: time series analysis
 * @return 0 on success, -1 on failure
 */
int stats_time_series_analysis(const double* timestamps, const double* performance_values,
                              int size, TimeSeriesAnalysis* analysis);

/**
 * @brief Detect change points in performance data
 * @param data Array of performance measurements
 * @param size Number of measurements
 * @param change_points Output: detected change points
 * @param max_change_points Maximum number of change points to detect
 * @return Number of change points detected
 */
int stats_detect_change_points(const double* data, int size, int* change_points,
                              int max_change_points);

/**
 * @brief Calculate moving average for trend smoothing
 * @param data Input time series data
 * @param size Number of data points
 * @param window_size Moving average window size
 * @param smoothed_data Output: smoothed data
 * @return 0 on success, -1 on failure
 */
int stats_moving_average(const double* data, int size, int window_size, double* smoothed_data);

// =====================================================
// Power Analysis and Sample Size
// =====================================================

/**
 * @brief Calculate statistical power for performance comparison
 * @param effect_size Expected effect size (Cohen's d)
 * @param sample_size1 Size of first group
 * @param sample_size2 Size of second group
 * @param alpha Significance level (typically 0.05)
 * @return Statistical power (0.0 to 1.0)
 */
double stats_calculate_power(double effect_size, int sample_size1, int sample_size2, 
                            double alpha);

/**
 * @brief Recommend sample size for desired statistical power
 * @param effect_size Expected effect size
 * @param desired_power Desired statistical power (typically 0.80)
 * @param alpha Significance level (typically 0.05)
 * @return Recommended sample size per group
 */
int stats_recommend_sample_size(double effect_size, double desired_power, double alpha);

/**
 * @brief Calculate minimum detectable effect size
 * @param sample_size1 Size of first group
 * @param sample_size2 Size of second group
 * @param power Statistical power
 * @param alpha Significance level
 * @return Minimum detectable effect size
 */
double stats_minimum_effect_size(int sample_size1, int sample_size2, double power, 
                                double alpha);

// =====================================================
// Advanced Diagnostics
// =====================================================

/**
 * @brief Perform comprehensive model diagnostics
 * @param residuals Array of residuals from model
 * @param fitted_values Array of fitted values
 * @param size Number of values
 * @param diagnostics_report Output: diagnostics report
 * @param buffer_size Size of diagnostics report buffer
 * @return 0 on success, -1 on failure
 */
int stats_model_diagnostics(const double* residuals, const double* fitted_values, int size,
                           char* diagnostics_report, size_t buffer_size);

/**
 * @brief Test for homoscedasticity (constant variance)
 * @param residuals Array of residuals
 * @param fitted_values Array of fitted values
 * @param size Number of values
 * @param p_value Output: p-value of Breusch-Pagan test
 * @return 1 if homoscedastic, 0 if heteroscedastic, -1 on error
 */
int stats_test_homoscedasticity(const double* residuals, const double* fitted_values, 
                               int size, double* p_value);

/**
 * @brief Bootstrap confidence intervals
 * @param data Original dataset
 * @param size Size of dataset
 * @param statistic Function to calculate statistic of interest
 * @param num_bootstrap Number of bootstrap samples
 * @param confidence_level Confidence level (e.g., 0.95)
 * @param ci_lower Output: lower bound of CI
 * @param ci_upper Output: upper bound of CI
 * @return 0 on success, -1 on failure
 */
int stats_bootstrap_ci(const double* data, int size, double (*statistic)(const double*, int),
                      int num_bootstrap, double confidence_level, 
                      double* ci_lower, double* ci_upper);

// =====================================================
// Reporting and Visualization
// =====================================================

/**
 * @brief Generate comprehensive statistical report
 * @param summary Statistical summary
 * @param comparisons Array of performance comparisons
 * @param num_comparisons Number of comparisons
 * @param report_buffer Output: formatted report
 * @param buffer_size Size of report buffer
 * @return 0 on success, -1 on failure
 */
int stats_generate_report(const StatisticalSummary* summary, 
                         const PerformanceComparison* comparisons, int num_comparisons,
                         char* report_buffer, size_t buffer_size);

/**
 * @brief Export statistical analysis to JSON
 * @param summary Statistical summary
 * @param comparisons Performance comparisons
 * @param num_comparisons Number of comparisons
 * @param output_file Output JSON file path
 * @return 0 on success, -1 on failure
 */
int stats_export_json(const StatisticalSummary* summary, 
                     const PerformanceComparison* comparisons, int num_comparisons,
                     const char* output_file);

/**
 * @brief Print statistical summary in formatted table
 * @param summary Statistical summary to print
 * @param label Label for the data being summarized
 */
void stats_print_summary(const StatisticalSummary* summary, const char* label);

/**
 * @brief Print performance comparison results
 * @param comparison Performance comparison results
 */
void stats_print_comparison(const PerformanceComparison* comparison);

// =====================================================
// Utility Functions
// =====================================================

/**
 * @brief Calculate basic statistics (mean, std, etc.)
 * @param data Array of values
 * @param size Number of values
 * @param mean Output: mean value
 * @param std_dev Output: standard deviation
 * @param min_val Output: minimum value
 * @param max_val Output: maximum value
 * @return 0 on success, -1 on failure
 */
int stats_basic_statistics(const double* data, int size, double* mean, double* std_dev,
                          double* min_val, double* max_val);

/**
 * @brief Sort array of doubles (for percentile calculations)
 * @param data Array to sort (modified in place)
 * @param size Number of elements
 */
void stats_sort_array(double* data, int size);

/**
 * @brief Calculate percentile value
 * @param sorted_data Sorted array of values
 * @param size Number of values
 * @param percentile Percentile to calculate (0.0 to 1.0)
 * @return Percentile value
 */
double stats_percentile(const double* sorted_data, int size, double percentile);

/**
 * @brief Validate statistical analysis parameters
 * @param data Data array
 * @param size Data size
 * @param min_required_size Minimum required sample size
 * @return 1 if valid, 0 if invalid
 */
int stats_validate_parameters(const double* data, int size, int min_required_size);

#ifdef __cplusplus
}
#endif