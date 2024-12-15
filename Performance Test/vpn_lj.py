import subprocess
import json
import csv

def save_results(protocol_name, results):
    # Save the ping and jitter results to the CSV file
    with open(f'{protocol_name}_vpn_results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            protocol_name,
            results.get('ping'),
            results.get('jitter')
        ])

def official_speedtest_test_with_server(protocol):
    for i in range(50):
        print(f'Running test {i+1}...')
        cmd = ['librespeed-cli', '--server', '91', '--no-download', '--no-upload', '--json']
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f'Test error: {result.stderr}')
            continue  # Skip this test

        output = result.stdout.strip()

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            print(f'Failed to parse JSON: {e}')
            print(f'Output: {output}')
            continue  # Skip this test

        # Ensure data is a non-empty list
        if isinstance(data, list) and len(data) > 0:
            test_result = data[0]
        else:
            print('Test result format incorrect, unable to parse.')
            continue  # Skip this test

        # Save results to CSV
        save_results(protocol, test_result)

        # Output test results
        ping_result = test_result.get('ping')
        jitter = test_result.get('jitter')

        print(f"Test {i+1} results:")
        print(f"  Ping: {ping_result:.2f} ms")
        print(f"  Jitter: {jitter:.2f} ms")
        print('-------------------------------------')

if __name__ == '__main__':
    protocol = 'nonvpn'

    # Run the tests
    official_speedtest_test_with_server(protocol)
