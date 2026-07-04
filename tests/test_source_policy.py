import unittest

from rag_agent.source_policy import (
    SourcePolicyError,
    authority_tier,
    is_allowed_hostname,
    validate_source_url,
)


class SourcePolicyTests(unittest.TestCase):
    def test_allows_exact_hosts_and_real_subdomains(self):
        self.assertTrue(is_allowed_hostname("utdallas.edu"))
        self.assertTrue(is_allowed_hostname("catalog.utdallas.edu"))
        self.assertTrue(is_allowed_hostname("WWW.UTSYSTEM.EDU."))

    def test_rejects_suffix_and_prefix_tricks(self):
        for host in (
            "utdallas.edu.evil.example",
            "fakeutdallas.edu",
            "utdallas.edu@example.com",
            "reddit.com",
            "policy.utdallas.edu.attacker.test",
        ):
            with self.subTest(host=host):
                self.assertFalse(is_allowed_hostname(host))

    def test_url_policy_is_https_default_deny(self):
        self.assertEqual(
            validate_source_url("https://catalog.utdallas.edu/2025/undergraduate/home/"),
            "https://catalog.utdallas.edu/2025/undergraduate/home/",
        )
        for url in (
            "http://utdallas.edu/page",
            "https://example.com/page",
            "https://utdallas.edu/login?next=/policy",
            "https://user:password@utdallas.edu/page",
        ):
            with self.subTest(url=url), self.assertRaises(SourcePolicyError):
                validate_source_url(url)

    def test_authority_tiers(self):
        self.assertEqual(authority_tier("https://www.utsystem.edu/board-of-regents/rules"), 1)
        self.assertEqual(authority_tier("https://policy.utdallas.edu/utdbp3102"), 2)
        self.assertEqual(authority_tier("https://catalog.utdallas.edu/2025/undergraduate"), 3)
        self.assertEqual(authority_tier("https://registrar.utdallas.edu/registration/"), 4)
        self.assertEqual(authority_tier("https://engineering.utdallas.edu/programs/"), 5)


if __name__ == "__main__":
    unittest.main()
