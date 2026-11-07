<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <rules key="{root}">
      <rule key="{r0}" label="Not in operation / reserve" symbol="0" filter="&quot;in_operation&quot; = 0" scalemaxdenom="1200000" scalemindenom="1" />
      <rule key="{r1}" label="Nuclear" symbol="1" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'nuclear'" />
      <rule key="{r2}" label="Lignite" symbol="2" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'lignite'" />
      <rule key="{r3}" label="Hard coal" symbol="3" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'hard_coal'" />
      <rule key="{r4}" label="Gas" symbol="4" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'gas'" />
      <rule key="{r5}" label="Oil" symbol="5" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'oil'" />
      <rule key="{r6}" label="Hydro (run-of-river, reservoir)" symbol="6" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'hydro'" />
      <rule key="{r7}" label="Pumped storage" symbol="7" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'pumped_storage'" />
      <rule key="{r8}" label="Battery and other storage" symbol="8" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'storage'" />
      <rule key="{r9}" label="Biomass" symbol="9" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'biomass'" />
      <rule key="{r10}" label="Waste" symbol="10" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'waste'" />
      <rule key="{r11}" label="Other" symbol="11" filter="&quot;in_operation&quot; = 1 AND &quot;display&quot; = 'individual' AND &quot;fuel_category&quot; = 'other'" />
    </rules>
    <symbols>
      <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="circle" />
            <Option name="color" type="QString" value="255,255,255,153" />
            <Option name="outline_color" type="QString" value="158,158,158,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="1" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="circle" />
            <Option name="color" type="QString" value="142,36,170,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="N" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="2" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="square" />
            <Option name="color" type="QString" value="121,85,72,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="L" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="3" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="square" />
            <Option name="color" type="QString" value="55,71,79,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="C" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="4" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="triangle" />
            <Option name="color" type="QString" value="239,108,0,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="G" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="5" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="triangle" />
            <Option name="color" type="QString" value="130,119,23,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="O" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="6" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="hexagon" />
            <Option name="color" type="QString" value="30,136,229,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="H" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="7" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="hexagon" />
            <Option name="color" type="QString" value="13,71,161,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="P" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="8" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="diamond" />
            <Option name="color" type="QString" value="0,137,123,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="S" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="9" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="circle" />
            <Option name="color" type="QString" value="124,179,66,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="B" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="10" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="diamond" />
            <Option name="color" type="QString" value="141,110,99,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="W" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="marker" name="11" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="name" type="QString" value="circle" />
            <Option name="color" type="QString" value="158,158,158,255" />
            <Option name="outline_color" type="QString" value="255,255,255,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.3" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="size" type="QString" value="2.0" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="miter" />
            <Option name="scale_method" type="QString" value="diameter" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
        <layer class="FontMarker" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="chr" type="QString" value="X" />
            <Option name="font" type="QString" value="Sans Serif" />
            <Option name="font_style" type="QString" value="Bold" />
            <Option name="color" type="QString" value="255,255,255,255" />
            <Option name="size" type="QString" value="1.4" />
            <Option name="size_unit" type="QString" value="MM" />
            <Option name="outline_width" type="QString" value="0" />
            <Option name="angle" type="QString" value="0" />
            <Option name="offset" type="QString" value="0,0" />
            <Option name="offset_unit" type="QString" value="MM" />
            <Option name="vertical_anchor_point" type="QString" value="1" />
            <Option name="horizontal_anchor_point" type="QString" value="1" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map">
                <Option name="size" type="Map">
                  <Option name="active" type="bool" value="true" />
                  <Option name="expression" type="QString" value="(max(1.8, least(1.2 + 0.16 * sqrt(coalesce(&quot;capacity_mw&quot;, 0)), 8.0))) * 0.55" />
                  <Option name="type" type="int" value="3" />
                </Option>
              </Option>
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="rule-based">
    <rules key="{labels-root}">
      <rule key="{label-0}" description="Plant name and MW" scalemaxdenom="2200000" scalemindenom="1" filter="&quot;in_operation&quot; = 1 AND &quot;capacity_mw&quot; &gt;= 300">
        <settings calloutType="simple">
          <text-style fieldName="&quot;name&quot; || ' (' || round(&quot;capacity_mw&quot;) || ' MW)'" isExpression="1" fontFamily="Sans Serif" fontSize="7.0" fontSizeUnit="Point" fontWeight="50" textColor="33,33,33,255" textOpacity="1" fontItalic="0" namedStyle="Regular" multilineHeight="1" allowHtml="0">
            <text-buffer bufferDraw="1" bufferSize="0.8" bufferSizeUnits="MM" bufferColor="255,255,255,230" bufferOpacity="0.9" bufferJoinStyle="128" bufferNoFill="1" />
          </text-style>
          <text-format wrapChar="" autoWrapLength="0" multilineAlign="0" plussign="0" reverseDirectionSymbol="0" />
          <placement placement="6" placementFlags="10" dist="1.2" distUnits="MM" priority="5" repeatDistance="0" overrunDistance="0" offsetType="0" quadOffset="4" centroidInside="1" fitInPolygonOnly="0" />
          <rendering obstacle="1" obstacleFactor="1" obstacleType="1" upsidedownLabels="0" displayAll="0" mergeLines="0" labelPerPart="0" limitNumLabels="0" maxNumLabels="2000" fontLimitPixelSize="0" scaleVisibility="0" drawLabels="1" />
        </settings>
      </rule>
    </rules>
  </labeling>
  <layerGeometryType>0</layerGeometryType>
</qgis>
