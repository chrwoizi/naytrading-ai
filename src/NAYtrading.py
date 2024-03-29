import requests
import json


class NAYtrading:

    def __init__(self, proxy_url, proxy_user, proxy_password, naytrading_url):
        if len(proxy_url) > 0:
            if len(proxy_user) > 0:
                self.proxies = {
                    'http': proxy_url % (proxy_user, proxy_password)
                }
            else:
                self.proxies = {
                    'http': proxy_url
                }
        else:
            self.proxies = None

        self.naytrading_url = naytrading_url

        self.session = requests.Session()

        self.jwt = None

    def login(self, naytrading_user, naytrading_password):
        url = self.naytrading_url + '/api/login'

        r = self.session.post(url, {
            'email': naytrading_user,
            'password': naytrading_password
        }, proxies=self.proxies)

        if r.status_code == 404:
            return Exception('Could not log in at naytrading (NOT FOUND)')

        if r.status_code != 200:
            raise Exception('%s returned %d' % (url, r.status_code))

        response = r.json()
        if response is None or response['token'] is None or len(response['token']) == 0:
            raise Exception('Could not log in at naytrading')

        self.jwt = response['token']

    def new_snapshot(self, max_age):
        url = self.naytrading_url + '/api/snapshot/new/random'
        if max_age > 0:
            url = url + '?max_age=' + str(max_age)
        r = self.session.get(url, proxies=self.proxies, timeout=120, headers={
                             "Authorization": "Bearer " + self.jwt})

        if r.status_code == 404:
            return None

        if r.status_code != 200:
            raise Exception('%s returned %d' % (url, r.status_code))

        data = r.json()
        if not data is None and data['snapshot']:
            data = data['snapshot']
        return data

    def new_ordered_snapshots(self, count):
        url = self.naytrading_url + '/api/snapshot/new/open'
        if count > 0:
            url = url + '?count=' + str(count)
        r = self.session.get(url, proxies=self.proxies, timeout=120, headers={
                             "Authorization": "Bearer " + self.jwt})

        if r.status_code == 404:
            return None

        if r.status_code != 200:
            raise Exception('%s returned %d' % (url, r.status_code))

        data = r.json()
        return data

    def set_decision(self, snapshot_id, decision):
        url = self.naytrading_url + '/api/decision'

        r = self.session.post(url, json.dumps({
            'id': snapshot_id,
            'decision': decision
        }), proxies=self.proxies, timeout=30, headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.jwt})

        if r.status_code != 200:
            raise Exception('%s returned %d' % (url, r.status_code))

        data = r.json()

        if 'status' not in data:
            raise Exception('%s returned no status' % (url))

        if data['status'] != 'ok':
            raise Exception('%s returned status %d' % (url, data['status']))

    def export_snapshots(self, from_date, file_path, report_progress):
        url = self.naytrading_url + '/api/export/user/snapshots/' + \
            from_date.strftime('%Y%m%d%H%M%S')
        r = self.session.get(url, proxies=self.proxies, timeout=600, stream=True, headers={
                             "Authorization": "Bearer " + self.jwt})

        if r.status_code != 200:
            raise Exception('%s returned %d' % (url, r.status_code))

        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    report_progress(f.tell())
                    f.write(chunk)

    def count_snapshots(self, from_date):
        url = self.naytrading_url + '/api/count/snapshots/' + \
            from_date.strftime('%Y%m%d%H%M%S')
        r = self.session.get(url, proxies=self.proxies, timeout=600, headers={
                             "Authorization": "Bearer " + self.jwt})

        if r.status_code != 200:
            raise Exception('%s returned %d' % (url, r.status_code))

        return int(r.text)
