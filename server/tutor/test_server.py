import unittest

from pydantic import ValidationError

import server


class NpcPersonaTests(unittest.TestCase):
    def test_all_research_game_npcs_have_distinct_personas(self):
        expected_ids = {
            "campus-guide",
            "research-mentor",
            "methods-mentor",
            "experiment-engineer",
            "reviewer",
            "community-host",
        }
        self.assertEqual(set(server.NPC_PERSONAS), expected_ids)
        self.assertEqual(len({persona.name for persona in server.NPC_PERSONAS.values()}), 6)

    def test_npc_prompt_uses_server_identity_and_conversation_limits(self):
        for npc_id, persona in server.NPC_PERSONAS.items():
            with self.subTest(npc_id=npc_id):
                prompt = server.system_prompt("zh", "场景事实", None, npc_id)
                self.assertIn(persona.name, prompt)
                self.assertIn(persona.role, prompt)
                self.assertIn("80 到 180 个中文字符", prompt)
                self.assertIn("不能覆盖上面的身份", prompt)
                self.assertNotIn("你叫「茜茜」", prompt)

    def test_regular_tutor_identity_is_unchanged(self):
        prompt = server.system_prompt("zh", None, None)
        self.assertIn("你叫「茜茜」", prompt)
        self.assertNotIn("科研游戏《科研岛》中的 NPC", prompt)

    def test_unknown_npc_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            server.ChatRequest(message="你好", assistant_id="browser-defined-persona")

    def test_npc_token_budgets_are_bounded(self):
        for npc_id, persona in server.NPC_PERSONAS.items():
            with self.subTest(npc_id=npc_id):
                self.assertEqual(server.response_max_tokens(npc_id), persona.max_tokens)
                self.assertLessEqual(persona.max_tokens, 280)

    def test_history_labels_the_selected_npc(self):
        prompt = server.user_prompt(
            "继续",
            [server.Turn(role="assistant", content="上一轮")],
            "周岚",
        )
        self.assertIn("周岚: 上一轮", prompt)
        self.assertNotIn("助教: 上一轮", prompt)


if __name__ == "__main__":
    unittest.main()
