import React from "react";
import { useTranslation } from "react-i18next";
import { Trash2 } from "lucide-react";
import type { ProAppRule, ProMatchType } from "@/bindings";
import { Input } from "../../ui/Input";
import { Button } from "../../ui/Button";
import { Dropdown } from "../../ui/Dropdown";
import { useSettings } from "../../../hooks/useSettings";

/**
 * The ordered app -> profile rules table. First enabled rule whose process/title substring
 * matches the foreground app wins; the default profile is used when none match.
 */
export const ProAppRulesEditor: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const rules = getSetting("pro_app_rules") || [];
  const profiles = getSetting("pro_profiles") || [];
  const updating = isUpdating("pro_app_rules");

  const persist = (next: ProAppRule[]) => updateSetting("pro_app_rules", next);
  const patchRule = (id: string, patch: Partial<ProAppRule>) =>
    persist(
      rules.map((rule) => (rule.id === id ? { ...rule, ...patch } : rule)),
    );
  const removeRule = (id: string) =>
    persist(rules.filter((rule) => rule.id !== id));
  const addRule = () =>
    persist([
      ...rules,
      {
        id: crypto.randomUUID(),
        match_type: "process",
        pattern: "",
        profile_key: profiles[0]?.key ?? "generic",
        enabled: true,
      },
    ]);

  return (
    <div className="px-4 py-3 space-y-3">
      {rules.length === 0 ? (
        <p className="text-sm text-mid-gray">
          {t("settings.postProcessing.pro.rules.empty")}
        </p>
      ) : (
        <div className="space-y-2">
          {rules.map((rule) => (
            <div key={rule.id} className="flex flex-wrap items-center gap-2">
              <input
                type="checkbox"
                className="w-4 h-4 accent-logo-primary cursor-pointer shrink-0"
                checked={rule.enabled ?? true}
                disabled={updating}
                onChange={(e) =>
                  patchRule(rule.id, { enabled: e.target.checked })
                }
                aria-label={t("settings.postProcessing.pro.rules.toggleAria")}
              />
              <Dropdown
                selectedValue={rule.match_type ?? "process"}
                options={[
                  {
                    value: "process",
                    label: t("settings.postProcessing.pro.rules.matchProcess"),
                  },
                  {
                    value: "title",
                    label: t("settings.postProcessing.pro.rules.matchTitle"),
                  },
                ]}
                onSelect={(value) =>
                  patchRule(rule.id, { match_type: value as ProMatchType })
                }
                disabled={updating}
                minWidth={0}
                className="w-32 shrink-0"
              />
              <Input
                variant="compact"
                value={rule.pattern}
                disabled={updating}
                onChange={(e) =>
                  patchRule(rule.id, { pattern: e.target.value })
                }
                placeholder={t(
                  "settings.postProcessing.pro.rules.patternPlaceholder",
                )}
                className="flex-1 min-w-[120px]"
              />
              <span className="text-mid-gray shrink-0" aria-hidden>
                →
              </span>
              <Dropdown
                selectedValue={rule.profile_key}
                options={profiles.map((profile) => ({
                  value: profile.key,
                  label: profile.label,
                }))}
                onSelect={(value) =>
                  patchRule(rule.id, { profile_key: value })
                }
                disabled={updating}
                minWidth={0}
                className="w-44 shrink-0"
              />
              <Button
                variant="danger-ghost"
                size="sm"
                disabled={updating}
                onClick={() => removeRule(rule.id)}
                aria-label={t("settings.postProcessing.pro.rules.deleteAria")}
                className="shrink-0"
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          ))}
        </div>
      )}
      <Button
        variant="secondary"
        size="sm"
        disabled={updating}
        onClick={addRule}
      >
        {t("settings.postProcessing.pro.rules.add")}
      </Button>
    </div>
  );
};
