def generate_recommendations(user_data, probability):
    recommendations = []

    # High monthly bill
    if user_data["MonthlyCharges"] > 80:
        recommendations.append(
            "Offer a discount or personalized pricing plan."
        )

    # Month-to-month customers
    if user_data["Contract"] == "Month-to-month":
        recommendations.append(
            "Promote an annual contract with additional benefits."
        )

    # New customers
    if user_data["tenure"] < 12:
        recommendations.append(
            "Assign a customer success representative for onboarding."
        )

    # Fiber customers
    if user_data["InternetService"] == "Fiber optic":
        recommendations.append(
            "Check service quality and provide premium support."
        )

    # Very high churn probability
    if probability > 0.80:
        recommendations.append(
            "Contact the customer immediately with a retention offer."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Customer is currently low risk. Continue regular engagement."
        )

    return recommendations