#include <cmath>
#include <functional>
#include <string>

#include <ignition/math/Color.hh>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/rendering/Scene.hh>
#include <gazebo/rendering/Visual.hh>

namespace gazebo
{
class TrafficLightLampPlugin : public VisualPlugin
{
 public:
  void Load(rendering::VisualPtr visual, sdf::ElementPtr sdf) override
  {
    visual_ = visual;
    role_ = ReadString(sdf, "role", "green");
    green_duration_ = ReadDuration(sdf, "green_duration", 15.0);
    yellow_duration_ = ReadDuration(sdf, "yellow_duration", 3.0);
    red_duration_ = ReadDuration(sdf, "red_duration", 10.0);
    cycle_duration_ = green_duration_ + yellow_duration_ + red_duration_;

    render_connection_ = event::Events::ConnectPreRender(
        std::bind(&TrafficLightLampPlugin::OnPreRender, this));

    gzmsg << "Traffic-light lamp plugin loaded: role=" << role_
          << ", visual=" << visual_->Name()
          << ", cycle=" << cycle_duration_ << "s" << std::endl;
  }

 private:
  enum class LightState
  {
    kGreen,
    kYellow,
    kRed
  };

  static std::string ReadString(const sdf::ElementPtr &sdf,
                                const std::string &element_name,
                                const std::string &default_value)
  {
    if (!sdf->HasElement(element_name))
    {
      return default_value;
    }
    return sdf->GetElement(element_name)->Get<std::string>();
  }

  static double ReadDuration(const sdf::ElementPtr &sdf,
                             const std::string &element_name,
                             double default_value)
  {
    if (!sdf->HasElement(element_name))
    {
      return default_value;
    }

    const double value = sdf->GetElement(element_name)->Get<double>();
    if (!std::isfinite(value) || value <= 0.0)
    {
      gzerr << "Invalid traffic-light " << element_name << " value; using "
            << default_value << "s" << std::endl;
      return default_value;
    }
    return value;
  }

  void OnPreRender()
  {
    if (!visual_)
    {
      return;
    }

    rendering::ScenePtr scene = visual_->GetScene();
    if (!scene)
    {
      return;
    }

    const LightState active_state = StateAt(scene->SimTime().Double());
    const bool lamp_on = IsActive(active_state);

    if (!initialized_ || lamp_on != lamp_on_)
    {
      SetLamp(lamp_on);
      lamp_on_ = lamp_on;
      initialized_ = true;
    }
  }

  LightState StateAt(double elapsed_seconds) const
  {
    double phase = std::fmod(elapsed_seconds, cycle_duration_);
    if (phase < 0.0)
    {
      phase += cycle_duration_;
    }

    if (phase < green_duration_)
    {
      return LightState::kGreen;
    }
    if (phase < green_duration_ + yellow_duration_)
    {
      return LightState::kYellow;
    }
    return LightState::kRed;
  }

  bool IsActive(LightState active_state) const
  {
    if (role_ == "red")
    {
      return active_state == LightState::kRed;
    }
    if (role_ == "yellow")
    {
      return active_state == LightState::kYellow;
    }
    return active_state == LightState::kGreen;
  }

  void SetLamp(bool enabled)
  {
    if (enabled)
    {
      ignition::math::Color color;
      if (role_ == "red")
      {
        color = ignition::math::Color(1.0, 0.02, 0.02, 1.0);
      }
      else if (role_ == "yellow")
      {
        color = ignition::math::Color(1.0, 0.65, 0.0, 1.0);
      }
      else
      {
        color = ignition::math::Color(0.05, 1.0, 0.05, 1.0);
      }

      visual_->SetAmbient(color);
      visual_->SetDiffuse(color);
      visual_->SetEmissive(color);
    }
    else
    {
      const ignition::math::Color off_color(0.05, 0.05, 0.05, 1.0);
      const ignition::math::Color no_emission(0.0, 0.0, 0.0, 1.0);
      visual_->SetAmbient(off_color);
      visual_->SetDiffuse(off_color);
      visual_->SetEmissive(no_emission);
    }
  }

  rendering::VisualPtr visual_;
  event::ConnectionPtr render_connection_;

  std::string role_ = "green";
  double green_duration_ = 15.0;
  double yellow_duration_ = 3.0;
  double red_duration_ = 10.0;
  double cycle_duration_ = 28.0;

  bool initialized_ = false;
  bool lamp_on_ = false;
};

GZ_REGISTER_VISUAL_PLUGIN(TrafficLightLampPlugin)
}  // namespace gazebo