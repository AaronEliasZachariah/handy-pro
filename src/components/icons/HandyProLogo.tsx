import React from "react";
import { useTranslation } from "react-i18next";
import HandyTextLogo from "./HandyTextLogo";

/**
 * handy-pro: the heritage "handy" wordmark with a "Pro" badge, used on the
 * first-run onboarding screens so new users see the full "Handy Pro" name
 * without redrawing the original hand-lettered SVG.
 */
const HandyProLogo = ({
  width = 200,
  className,
}: {
  width?: number;
  className?: string;
}) => {
  const { t } = useTranslation();
  return (
    <div className={`flex items-end gap-2 ${className ?? ""}`}>
      <HandyTextLogo width={width} />
      <span
        className="logo-primary font-semibold leading-none tracking-tight text-2xl pb-[0.35em]"
        aria-hidden="true"
      >
        {t("onboarding.proBadge")}
      </span>
    </div>
  );
};

export default HandyProLogo;
