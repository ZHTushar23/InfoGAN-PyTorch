import torch
import torch

def quantize(value, levels):
    """Quantize a normalized value into a given number of levels."""
    step = 1.0 / levels
    return min(int(value / step), levels - 1)

def sza_vza_to_code(sza, vza):
    """
    Convert SZA and VZA to a 5-bit code.
    
    Parameters:
    sza (float): Solar Zenith Angle (0 to 180 degrees)
    vza (float): View Zenith Angle (-180 to 180 degrees)
    
    Returns:
    str: 5-bit code
    """
    # Normalize SZA to the range [0, 1]
    sza_normalized = sza / 180.0
    
    # Normalize VZA from the range [-180, 180] to [0, 1]
    vza_normalized = (vza + 180.0) / 360.0
    
    # Quantize SZA into 4 levels (2 bits)
    sza_quantized = quantize(sza_normalized, 4)
    # Quantize VZA into 8 levels (3 bits)
    vza_quantized = quantize(vza_normalized, 8)
    
    # Convert to binary strings and pad with zeros if necessary
    sza_bits = format(sza_quantized, '02b')
    vza_bits = format(vza_quantized, '03b')
    
    # Concatenate the bits to form the 5-bit code
    code = sza_bits + vza_bits
    
    return code

# Example usage
sza = 4.0  # Example SZA
vza = 30.0  # Example VZA

code = sza_vza_to_code(sza, vza)
print(f"SZA: {sza}, VZA: {vza}, Code: {code}")
