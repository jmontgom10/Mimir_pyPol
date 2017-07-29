"""
Provides the appropriate header_handler for the Mimir instrument.
"""

def Mimir_header_handler(header):
    """Makes some small modifications to the header as its read in"""
    # Copy the header for manipulation
    outHeader = header.copy()

    # Make the ADU units lower case so that astropy.units can handle them.
    outHeader['BUNIT'] = header['BUNIT'].strip().lower()

    # Set the gain for the instrument (since it wasn't included in the header)
    #
    # http://people.bu.edu/clemens/mimir/capabilties.html
    #
    # on July 29, 2017
    outHeader['AGAIN_01'] = 8.21
    outHeader['ARDNS_01'] = 17.8

    return outHeader
