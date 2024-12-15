import subprocess
import csv
import os
import json

def run_iperf3_upload(protocol_name, server_ip, duration=30):
    print(f'Starting upload bandwidth test for {protocol_name}...')
    result = subprocess.run(
        ['iperf3', '-c', server_ip, '-t', str(duration), '-J'],  # JSON output
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(f'Error running iperf3 (upload): {result.stderr.decode()}')
        return None

    output = result.stdout.decode()

    try:
        data = json.loads(output)
        upload_bandwidth_data = extract_bandwidth_from_intervals(data)

        if upload_bandwidth_data:
            print(f'Upload test completed for {protocol_name}.')
            return {
                'Protocol': protocol_name,
                'Bandwidth Rates (Mbps)': upload_bandwidth_data['rates'],
                'Max Bandwidth (Mbps)': upload_bandwidth_data['max'],
                'Min Bandwidth (Mbps)': upload_bandwidth_data['min'],
                'Avg Bandwidth (Mbps)': upload_bandwidth_data['avg']
            }
        else:
            print('Error parsing iperf3 upload output.')
            return None
    except json.JSONDecodeError as e:
        print(f'Error decoding JSON (upload): {e}')
        return None

def run_iperf3_download(protocol_name, server_ip, duration=30):
    print(f'Starting download bandwidth test for {protocol_name}...')
    result = subprocess.run(
        ['iperf3', '-c', server_ip, '-t', str(duration), '-J', '-R'],  # Reverse mode for download, JSON output
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(f'Error running iperf3 (download): {result.stderr.decode()}')
        return None

    output = result.stdout.decode()

    try:
        data = json.loads(output)
        download_bandwidth_data = extract_bandwidth_from_intervals(data)

        if download_bandwidth_data:
            print(f'Download test completed for {protocol_name}.')
            return {
                'Protocol': protocol_name,
                'Bandwidth Rates (Mbps)': download_bandwidth_data['rates'],
                'Max Bandwidth (Mbps)': download_bandwidth_data['max'],
                'Min Bandwidth (Mbps)': download_bandwidth_data['min'],
                'Avg Bandwidth (Mbps)': download_bandwidth_data['avg']
            }
        else:
            print('Error parsing iperf3 download output.')
            return None
    except json.JSONDecodeError as e:
        print(f'Error decoding JSON (download): {e}')
        return None

def extract_bandwidth_from_intervals(data):
    try:
        interval_bandwidths = [
            interval['sum']['bits_per_second'] / 1e6  # Convert to Mbps
            for interval in data['intervals']
        ]

        if interval_bandwidths:
            return {
                'rates': interval_bandwidths,
                'max': max(interval_bandwidths),
                'min': min(interval_bandwidths),
                'avg': sum(interval_bandwidths) / len(interval_bandwidths)
            }
    except KeyError as e:
        print(f'Error extracting bandwidth data from intervals: {e}')
    return None

def save_to_csv(data, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(['Protocol', 'Rate Index', 'Bandwidth (Mbps)'])

        # Save interval bandwidth data
        for idx, rate in enumerate(data['Bandwidth Rates (Mbps)'], start=1):
            writer.writerow([data['Protocol'], idx, rate])

        # Save summary data
        writer.writerow([data['Protocol'], 'Summary', f"Max: {data['Max Bandwidth (Mbps)']}, Min: {data['Min Bandwidth (Mbps)']}, Avg: {data['Avg Bandwidth (Mbps)']}"])

if __name__ == '__main__':
    protocol_name = 'wireguard'  # Replace with the actual protocol name you are testing
    server_ip = '13.52.247.161'  # Replace with actual IP address

    # Run iperf3 to test upload bandwidth
    upload_bandwidth_data = run_iperf3_upload(protocol_name, server_ip)
    if upload_bandwidth_data:
        save_to_csv(upload_bandwidth_data, 'bandwidth/wireguard_upload.csv')
        print('Upload bandwidth data has been saved to upload_bandwidth_test.csv.')

    # Run iperf3 to test download bandwidth
    download_bandwidth_data = run_iperf3_download(protocol_name, server_ip)
    if download_bandwidth_data:
        save_to_csv(download_bandwidth_data, 'bandwidth/wireguard_download.csv')
        print('Download bandwidth data has been saved to download_bandwidth_test.csv.')
