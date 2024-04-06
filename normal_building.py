
import streamlit as st
import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

def process_image(image, min_val, max_val, mode, method, kernel_size):
    # Ensure kernel size is odd
    kernel_size = kernel_size * 2 + 1

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Convert mode and method to OpenCV values
    mode_value = getattr(cv2, mode)
    method_value = getattr(cv2, method)

    # Canny edge detection
    edges = cv2.Canny(blurred, min_val, max_val)

    # Find contours
    contours, _ = cv2.findContours(edges, mode_value, method_value)

    # Draw contours on a blank image
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

    # Convert images from BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

    # Return processed images
    return image_rgb, gray, blurred, edges, contour_image_rgb, contours

def inverted_gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def fit_gaussian_to_contour(contours):
    """Fits a Gaussian to the largest contour."""
    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    all_x = []
    all_y = []

    # Loop through all contours
    for contour in contours:
        # Extract x and y coordinates from the contour
        x = contour[:, 0, 0]
        y = contour[:, 0, 1]
    
        # Append to the lists
        all_x.extend(x)
        all_y.extend(y)
    
    # Convert lists to arrays for plotting
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Assuming 'all_x' and 'all_y_inverted' are the numpy arrays containing your contour data:
    points = np.column_stack((all_x, all_y))  # Combine x and y coordinates

    filtered_points = points[points[:, 1] <= 468]

    # Separate the filtered x and y coordinates
    filtered_x = filtered_points[:, 0]
    filtered_y = filtered_points[:, 1]
    
    # Sort the filtered_x and corresponding filtered_y for fitting
    sorted_indices = np.argsort(filtered_x)
    sorted_x = filtered_x[sorted_indices]
    sorted_y = filtered_y[sorted_indices]
    
    # Fit the inverted Gaussian function to the data
    initial_guess = [min(sorted_y), max(sorted_y) - min(sorted_y), np.mean(sorted_x), np.std(sorted_x)]
    parameters, covariance = curve_fit(inverted_gauss, sorted_x, sorted_y, p0=initial_guess)
    
    # Calculate the fitted y-values
    fit_y = inverted_gauss(sorted_x, *parameters)

    # Chi-squared per degree of freedom calculation
    residuals = sorted_y - fit_y
    sigma_value = 1  # Assuming 1 pixel uncertainty
    sigmas = np.full_like(residuals, sigma_value)
    chi_squared = np.sum((residuals / sigmas)**2)
    degrees_of_freedom = len(sorted_y) - len(parameters)
    chi_squared_per_dof = chi_squared / degrees_of_freedom

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sorted_x, sorted_y, s=1, label='Data')
    ax.plot(sorted_x, fit_y, color='red', label='Fit')
    ax.invert_yaxis()  # Invert y-axis if necessary for your data
    ax.legend()
    plt.title('Gaussian Fit for Extracted Contour')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

    # Annotation with fit parameters and chi-squared statistic
    annotation_text = (
        f"Fit parameters:\nH: {parameters[0]:.2f}\nA: {parameters[1]:.2f}\n"
        f"x0: {parameters[2]:.2f}\nsigma: {parameters[3]:.2f}\n"
        f"Reduced Chi-squared: {chi_squared_per_dof:.2f}"
    )
    plt.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', verticalalignment='top',
                 bbox=dict(boxstyle="round", alpha=0.5))

    return fig, parameters, chi_squared_per_dof  # Return the figure object for display in Streamlit

st.set_page_config(layout="centered")
st.title('Is this Building Normal?')

st.write(
    """
    This app analyzes building images to assess if their contours follow a Gaussian distribution, 
    indicating "normal" structural features. Adjust the image processing parameters on the left 
    to optimize contour detection before fitting.The fitting results are stored in a dataframe that you can download as a .csv.
    """
)
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.sidebar.header('Image Processing Parameters')

    min_val = st.sidebar.slider(
        'Min Threshold:', min_value=0, max_value=255, value=255, key='min_val',
        help='Minimum threshold for Canny edge detection. Increase to detect stronger edges.'
    )
    
    max_val = st.sidebar.slider(
        'Max Threshold:', min_value=0, max_value=255, value=109, key='max_val',
        help='Maximum threshold for Canny edge detection. Decrease to include softer edges.'
    )
    
    kernel_size = st.sidebar.slider(
        'Kernel Size:', min_value=0, max_value=10, value=3, key='kernel_size',
        help='Size of the Gaussian kernel used for blurring. Increase to smooth over more noise.'
    )
    
    mode = st.sidebar.selectbox('Contour Retrieval Mode:',
        ['RETR_EXTERNAL', 'RETR_LIST', 'RETR_CCOMP', 'RETR_TREE'], index=0, key='mode',
        help='Defines the contour retrieval approach. "RETR_EXTERNAL" retrieves only extreme outer contours.'
    )
    
    method = st.sidebar.selectbox('Contour Approximation Method:',
        ['CHAIN_APPROX_NONE', 'CHAIN_APPROX_SIMPLE', 'CHAIN_APPROX_TC89_L1', 'CHAIN_APPROX_TC89_KCOS'], index=1, key='method',
        help='Determines the contour approximation method. "CHAIN_APPROX_SIMPLE" compresses horizontal, vertical, and diagonal segments.'
    )
    # Process the image immediately after any input changes
    original_rgb, gray, blurred, edges, contours_rgb, contours = process_image(image, min_val, max_val, mode, method, kernel_size)

    # Display the images side by side using columns
    cols = st.columns(5)  # Create five columns for the five images
    with cols[0]:
        st.image(original_rgb, caption='Original Image', use_column_width=True)
    with cols[1]:
        st.image(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), caption='Grayscale', use_column_width=True)
    with cols[2]:
        st.image(cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB), caption='Blurred', use_column_width=True)
    with cols[3]:
        st.image(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), caption='Edges', use_column_width=True)
    with cols[4]:
        st.image(contours_rgb, caption='Contours', use_column_width=True)
    
    fig, parameters, chi_squared_per_dof = fit_gaussian_to_contour(contours)
    st.pyplot(fig)
    
    # Append the new fit results to the DataFrame
    new_row = {
        'H': parameters[0],
        'A': parameters[1],
        'x0': parameters[2],
        'sigma': parameters[3],
        'Reduced Chi-squared': chi_squared_per_dof
    }
    
    # Initialize the DataFrame in session state if it does not already exist
    if 'fit_results_df' not in st.session_state:
        st.session_state.fit_results_df = pd.DataFrame(columns=[
            'H', 'A', 'x0', 'sigma', 'Reduced Chi-squared',
            'min_val', 'max_val', 'kernel_size', 'mode', 'method'
        ])
        
    # Updating the new_row dictionary to include both sets of parameters
    new_row = {
        'H': parameters[0],
        'A': parameters[1],
        'x0': parameters[2],
        'sigma': parameters[3],
        'Reduced Chi-squared': chi_squared_per_dof,
        'min_val': min_val,
        'max_val': max_val,
        'kernel_size': kernel_size,
        'mode': mode,
        'method': method,
    }
    
    new_row_df = pd.DataFrame([new_row])
    
    # Use concat to add the new row to the DataFrame
    st.session_state.fit_results_df = pd.concat([st.session_state.fit_results_df, new_row_df], ignore_index=True)
    
    st.dataframe(st.session_state.fit_results_df)
    